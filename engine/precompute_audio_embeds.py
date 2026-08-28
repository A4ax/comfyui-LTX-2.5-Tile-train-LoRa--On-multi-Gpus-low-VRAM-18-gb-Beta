"""Precompute connector-applied video/audio embeds into each condition file.

Applied OFFLINE during dataset preprocessing (once), so the training engine does
NOT need to load the embeddings processor on the GPUs -> keeps training VRAM at
face-only levels (fixes the 12GB OOM in voice+face training).

For every `*.pt` under --conditions-dir it loads the raw `video_prompt_embeds` /
`audio_prompt_embeds` (4096-dim features) + `prompt_attention_mask`, runs the
embeddings processor's connectors, and adds `video_embeds` (4096) + `audio_embeds`
(2048) + `embeds_mask` keys back into the same file.
"""
import argparse
import os
import sys
import time

import engine_env  # noqa: E402
engine_env.setup_paths()
from msvc_env import apply_msvc_env  # noqa: E402
apply_msvc_env()

import torch  # noqa: E402

from load_times import record  # noqa: E402

from ltx_trainer.model_loader import load_embeddings_processor  # noqa: E402


def build_additive(mask, dtype, device):
    """Binary (L,) attention mask -> (1,1,1,L) additive mask (0 valid, -finfo.max pad)."""
    mask = mask.to(device=device, dtype=dtype)
    fm = torch.finfo(dtype).max
    additive = torch.where(mask > 0.5, torch.zeros_like(mask), torch.full_like(mask, -fm))
    return additive.unsqueeze(0).unsqueeze(0).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions-dir", required=True)
    ap.add_argument("--sidecar", default="", help="embeddings_processor_bf16.safetensors")
    ap.add_argument("--text-encoder", default="", help="packed Gemma text encoder")
    ap.add_argument("--device", default=None, help="target device (overridden by --gpus)")
    ap.add_argument("--gpus", default="", help="comma-separated GPU indices to place the processor on")
    ap.add_argument("--load-times", default="", help="load_times.jsonl path to append model load times")
    a = ap.parse_args()

    gpus = [int(x) for x in a.gpus.split(",") if x.strip() != ""]
    if gpus:
        device = torch.device(f"cuda:{gpus[0]}")
    else:
        device = torch.device(a.device or "cuda:0")
    dtype = torch.bfloat16

    if not os.path.isdir(a.conditions_dir):
        print(f"[precompute] conditions dir not found: {a.conditions_dir}", flush=True)
        sys.exit(1)

    print(f"[precompute] loading embeddings processor on {device}...", flush=True)
    _t = time.time()
    ep = load_embeddings_processor(a.sidecar, gemma_model_path=a.text_encoder, device=device)
    ep.to(device).eval()
    _dt = time.time() - _t
    record(a.load_times, "embeddings_processor", _dt)
    print(f"[precompute] embeddings processor loaded in {_dt:.1f}s", flush=True)

    files = []
    for root, _, names in os.walk(a.conditions_dir):
        for n in names:
            if n.endswith(".pt"):
                files.append(os.path.join(root, n))
    files.sort()
    print(f"[precompute] found {len(files)} condition files", flush=True)

    done = 0
    skipped = 0
    for path in files:
        try:
            cd = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(cd, dict):
                print(f"[precompute] skip (not a dict): {path}", flush=True)
                skipped += 1
                continue
            if cd.get("audio_embeds") is not None:
                # Fully processed (video + audio embeds already cached). Skip.
                skipped += 1
                continue
            if cd.get("video_embeds") is not None and cd.get("audio_prompt_embeds") is None:
                # Image-only clip: video embeds cached and there is no audio track to
                # backfill, so there is nothing left to compute. Skip.
                skipped += 1
                continue
            vfeats = cd.get("video_prompt_embeds")
            afeats = cd.get("audio_prompt_embeds")
            mask = cd.get("prompt_attention_mask")
            if vfeats is None or mask is None:
                skipped += 1
                continue
            vfeats = vfeats.unsqueeze(0).to(device=device, dtype=dtype)
            afeats_t = afeats.unsqueeze(0).to(device=device, dtype=dtype) if afeats is not None else None
            additive = build_additive(mask, dtype, device)
            with torch.no_grad():
                ve, ae, _ = ep.create_embeddings(vfeats, afeats_t, additive)
            cd["video_embeds"] = ve.cpu().contiguous()
            cd["audio_embeds"] = ae.cpu().contiguous() if ae is not None else None
            torch.save(cd, path)
            done += 1
        except Exception as e:
            print(f"[precompute] error {path}: {e}", flush=True)
    print(f"[precompute] DONE cached={done} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
