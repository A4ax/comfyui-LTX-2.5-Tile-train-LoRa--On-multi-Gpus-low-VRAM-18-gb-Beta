"""Encode captions -> context embeddings (ctx.pt) using the 8-bit Gemma, 100% on GPUs.

GPU-only (never materializes the full model on CPU RAM or one GPU):
  1. Build the Gemma skeleton on META (no weight data).
  2. Load non-layer bf16 weights straight onto cuda:0 via safe_open(device=...).
  3. For EACH transformer layer: read its bf16 weights straight onto a 3060,
     swap that layer to 8-bit (bnb Linear8bitLt) on GPU, then free its bf16.
     -> only one layer in bf16 at a time; 8-bit accumulates to ~13GB across the
        2x3060 (~6.5GB each); the 4060 is never needed.
  4. Swap the remaining non-layer linears to 8-bit on GPU.
  5. dispatch_model routes the cross-GPU forward.
  6. Encode all captions on GPU -> hidden states kept on GPU.
  7. Free the Gemma, load the embeddings processor on GPU, produce ctx.
  8. Write ctx_<hash>.pt + index.json, then free the GPUs (process exits).

Usage:
  python encode_captions.py --text-encoder <gemma.safetensors> --sidecar <emb.safetensors> \
      --captions "a; b" --out-dir <dir>
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import engine_env  # noqa: E402
engine_env.setup_paths()
from msvc_env import apply_msvc_env  # noqa: E402
apply_msvc_env()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import bitsandbytes as bnb  # noqa: E402
import safetensors  # noqa: E402

from load_times import record  # noqa: E402

from ltx_trainer.model_loader import load_embeddings_processor  # noqa: E402
from ltx_trainer import gemma_8bit  # noqa: E402
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder  # noqa: E402
from ltx_core.text_encoders.gemma import GemmaTextEncoderConfigurator, get_gemma_ops  # noqa: E402
from ltx_core.text_encoders.gemma.gemma_assets import resolve_gemma_weight_paths  # noqa: E402

LOAD_GPUS = [0, 1]          # two 3060s for the 8-bit model (~6.5GB each)
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def vram():
    return {g: round(torch.cuda.memory_allocated(g) / 1e9, 2) for g in [0, 1, 2]}


def load_gemma_8bit_gpu(te_path):
    """Meta skeleton -> per-layer bf16 load, 8-bit swap, CPU->GPU quantize.

    bnb 8-bit (LLM.int8) quantizes during the CPU->GPU move. To keep 8-bit memory
    (13GB on the 2x3060s) we do it ONE LAYER at a time: only a single layer's
    bf16 (~0.5GB) is ever in CPU RAM transiently for the quantize hop; the model
    accumulates as int8 on the GPUs and ALL compute stays on CUDA."""
    t0 = time.time()
    sd_ops, module_ops = get_gemma_ops(te_path)
    builder = SingleGPUModelBuilder(
        model_path=tuple(resolve_gemma_weight_paths(te_path)),
        model_class_configurator=GemmaTextEncoderConfigurator.with_gemma_model_path(te_path),
        model_sd_ops=sd_ops, module_ops=module_ops)
    te = builder.meta_model(builder.model_metadata(), module_ops)
    lm = te.model.model.language_model
    n_layers = len(lm.layers)
    print(f"[encode] meta skeleton, {n_layers} layers ({time.time()-t0:.0f}s)", flush=True)

    # split raw keys into per-layer vs non-layer (mapped param path)
    layers_raw = defaultdict(dict)   # layer_idx -> {mapped_rel_key: raw_key}
    nonlayer_raw = {}                # mapped_key -> raw_key
    with safetensors.safe_open(te_path, framework="pt") as f:
        for raw in f.keys():
            mapped = sd_ops.apply_to_key(raw)
            if mapped is None:
                continue
            m = LAYER_RE.search(mapped)
            if m:
                li = int(m.group(1))
                rel = mapped[mapped.index(f".layers.{li}.") + len(f".layers.{li}."):]
                layers_raw[li][rel] = raw
            elif mapped.startswith("model.model.language_model."):
                # only language_model non-layer (embed_tokens/norm) is needed for
                # text encode; vision/audio towers + multimodal projectors are
                # unused and stay meta (saves ~5.5GB and avoids the cuda:0 OOM)
                nonlayer_raw[mapped] = raw

    # 1) non-layer bf16 on CPU (small), assigned to skeleton
    with safetensors.safe_open(te_path, framework="pt") as f:
        sd0 = {mapped: f.get_tensor(raw).to(torch.bfloat16) for mapped, raw in nonlayer_raw.items()}
    te.load_state_dict(sd0, strict=False, assign=True)
    del sd0
    print(f"[encode] non-layer weights staged on CPU ({time.time()-t0:.0f}s)", flush=True)

    # 2) per-layer: bf16 CPU -> swap 8-bit -> .to(cuda) quantizes to int8 on GPU.
    #    Put ~40% of layers on cuda:0 and ~60% on cuda:1 so cuda:0 keeps headroom
    #    for dispatch_model's internal re-quantize (bnb .to() allocates transiently).
    def layer_gpu(li):
        return LOAD_GPUS[0] if (li % 5) < 2 else LOAD_GPUS[1]

    with safetensors.safe_open(te_path, framework="pt") as f:
        for li in range(n_layers):
            if li not in layers_raw:
                continue
            gpu = layer_gpu(li)
            layer_sd = {rel: f.get_tensor(raw).to(torch.bfloat16) for rel, raw in layers_raw[li].items()}
            layer_mod = lm.layers[li]
            layer_mod.load_state_dict(layer_sd, strict=False, assign=True)
            del layer_sd
            gemma_8bit._replace_linear_with_8bit(layer_mod, bnb)   # Int8Params fp16 on CPU
            layer_mod.to(f"cuda:{gpu}")                             # int8 quantize on GPU
            torch.cuda.empty_cache()
            if li % 8 == 7:
                print(f"[encode] layer {li+1}/{n_layers} 8-bit vram={vram()} ({time.time()-t0:.0f}s)", flush=True)

    # 3) swap remaining non-layer linears to 8-bit + move CPU params to the ODD 3060
    #    (keeps cuda:0 at only its layer share so the dispatch re-quantize has headroom)
    nonlayer_gpu = LOAD_GPUS[1]
    gemma_8bit._replace_linear_with_8bit(te.model, bnb)
    for name, p in list(te.model.named_parameters()):
        if p.device.type == "cpu":
            p.data = p.data.to(f"cuda:{nonlayer_gpu}")
    for name, b in list(te.model.named_buffers()):
        if b.device.type == "cpu":
            b.data = b.data.to(f"cuda:{nonlayer_gpu}")
    print(f"[encode] non-layer 8-bit + moved to cuda:{nonlayer_gpu} vram={vram()} ({time.time()-t0:.0f}s)", flush=True)

    # 4) cross-GPU routing: pre-hooks move each module's input onto its own device
    #    (avoids accelerate.dispatch_model, whose bnb re-quantize OOMs).
    def _move(x, dev):
        if isinstance(x, torch.Tensor):
            return x.to(dev) if x.device != dev else x
        if isinstance(x, tuple):
            return tuple(_move(e, dev) for e in x)
        if isinstance(x, list):
            return [_move(e, dev) for e in x]
        if isinstance(x, dict):
            return {k: _move(v, dev) for k, v in x.items()}
        return x

    def _route_hook(module, args, kwargs):
        dev = next(module.parameters()).device
        return tuple(_move(a, dev) for a in args), {k: _move(v, dev) for k, v in kwargs.items()}

    for m in [lm.embed_tokens, *lm.layers, lm.norm]:
        m.register_forward_pre_hook(_route_hook, with_kwargs=True)
    print(f"[encode] 8-bit Gemma on {LOAD_GPUS} with custom routing vram={vram()} ({time.time()-t0:.0f}s)", flush=True)
    return te


def main():
    torch.cuda.empty_cache()  # clear any residual GPU state before loading
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-encoder", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--captions", required=True, help="; separated captions")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpus", default="0,1", help="comma-separated GPU indices for the text encoder")
    ap.add_argument("--connectors-device", default=None, help="gpu0|gpu1|gpu2|cpu for the embeddings processor")
    ap.add_argument("--load-times", default="", help="load_times.jsonl path to append model load times")
    args = ap.parse_args()

    global LOAD_GPUS
    LOAD_GPUS = [int(x) for x in args.gpus.split(",") if x.strip() != ""] or [0, 1]

    captions = [c.strip() for c in args.captions.split(";") if c.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    _t = time.time()
    te = load_gemma_8bit_gpu(args.text_encoder)
    record(args.load_times, "text_encoder", time.time() - _t)

    hidden = []
    _dcmap = {"gpu0": "cuda:0", "gpu1": "cuda:1", "gpu2": "cuda:2", "cpu": "cpu"}
    proc_dev = _dcmap.get(args.connectors_device) or f"cuda:{LOAD_GPUS[0]}"
    with torch.inference_mode():
        for i, cap in enumerate(captions):
            print(f"[encode] [{i+1}/{len(captions)}] gemma encode (GPU)... {time.time()-t0:.0f}s", flush=True)
            hs, mask = te.encode([cap])[0]
            hs = tuple(h.to(proc_dev) for h in hs) if isinstance(hs, tuple) else hs.to(proc_dev)
            mask = mask.to(proc_dev) if mask is not None else None
            hidden.append((cap, hs, mask))
            print(f"[encode]   vram={vram()}", flush=True)

    del te
    import gc; gc.collect()
    torch.cuda.empty_cache()
    print(f"[encode] Gemma freed, vram={vram()} ({time.time()-t0:.0f}s)", flush=True)

    print(f"[encode] loading embeddings processor on {proc_dev}...", flush=True)
    _t = time.time()
    ep = load_embeddings_processor(args.sidecar, gemma_model_path=args.text_encoder,
                                   device=proc_dev, dtype=torch.bfloat16)
    record(args.load_times, "embeddings_processor", time.time() - _t)
    print(f"[encode] processor on GPU, vram={vram()} ({time.time()-t0:.0f}s)", flush=True)

    cond_dir = os.path.join(args.out_dir, "conditions")
    os.makedirs(cond_dir, exist_ok=True)
    index = []
    with torch.inference_mode():
        for cap, hs, mask in hidden:
            out = ep.process_hidden_states(hs, mask)
            ctx = out.video_encoding.to(torch.bfloat16).contiguous()
            audio_enc = getattr(out, "audio_encoding", None)
            if audio_enc is not None:
                audio_enc = audio_enc.to(torch.bfloat16).contiguous()
            h = hashlib.md5(cap.encode()).hexdigest()[:10]
            path = os.path.join(args.out_dir, f"ctx_{h}.pt")
            torch.save({"caption": cap, "ctx": ctx.cpu(),
                        "audio_encoding": audio_enc.cpu() if audio_enc is not None else None}, path)

            # trainer-compatible condition file (feature-extractor output; the
            # connector is applied during training) so the Dataset step can consume
            # this directly instead of running the 26GB Gemma on CPU.
            vfeats, afeats = ep.feature_extractor(hs, mask, "left")
            cond = {
                "video_prompt_embeds": vfeats[0].cpu().contiguous(),
                "prompt_attention_mask": mask[0].cpu().contiguous(),
            }
            if afeats is not None:
                cond["audio_prompt_embeds"] = afeats[0].cpu().contiguous()
            cond_path = os.path.join(cond_dir, f"{h}.pt")
            torch.save(cond, cond_path)

            index.append({"caption": cap, "hash": h, "ctx": path, "cond": cond_path,
                          "shape": list(ctx.shape), "dtype": str(ctx.dtype)})
            print(f"[encode]   -> {os.path.basename(path)} shape={tuple(ctx.shape)} cond={os.path.basename(cond_path)} ({time.time()-t0:.0f}s)", flush=True)

    with open(os.path.join(args.out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    del ep
    gc.collect(); torch.cuda.empty_cache()
    print(f"[encode] DONE {len(captions)} captions in {time.time()-t0:.0f}s; GPUs freed vram={vram()}", flush=True)
    print("[encode] PASS", flush=True)


if __name__ == "__main__":
    main()
