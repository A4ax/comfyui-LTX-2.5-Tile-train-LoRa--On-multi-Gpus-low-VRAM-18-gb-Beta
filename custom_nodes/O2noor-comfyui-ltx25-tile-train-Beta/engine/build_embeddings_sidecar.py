"""Build a loadable embeddings-processor sidecar (feature_extractor + connectors).

`load_embeddings_processor` builds a skeleton named `feature_extractor.*` +
`video_connector.*`/`audio_connector.*`. The pieces live split across two assets:
  - feature extractor: TEXT_ENC  `text_embedding_projection.{video,audio}_aggregate_embed.*`
  - connectors:        int8-convrot  `model.diffusion_model.{video,audio}_embeddings_connector.*`
                       (convrot int8 -> dequant to bf16)

This script writes ONE sidecar with the FINAL skeleton key names, so
`load_embeddings_processor(checkpoint_path=<sidecar>, gemma_model_path=TEXT_ENC)`
loads without remapping. Output: models/diffusion_models/embeddings_processor_bf16.safetensors
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import os  # noqa: E402
import engine_env  # noqa: E402
engine_env.setup_paths()
from msvc_env import apply_msvc_env
apply_msvc_env()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import json  # noqa: E402
import torch  # noqa: E402
import safetensors  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

_models = os.path.join(engine_env.MODELS_ROOT, "diffusion_models")
TEXT_ENC = engine_env.load_config().get("text_encoder") or os.path.join(_models, "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
INT8 = os.path.join(_models, "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors")
DST = os.path.join(_models, "embeddings_processor_bf16.safetensors")
PREFIX = "model.diffusion_model."
_device = "cuda:0"
HADAMARD_CACHE = {}


def build_hadamard(size):
    if size in HADAMARD_CACHE:
        return HADAMARD_CACHE[size]
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    h = h / (size ** 0.5)
    HADAMARD_CACHE[size] = h
    return h


def dequant_convrot(w_int8, w_scale, group_size):
    dev = _device if torch.cuda.is_available() else "cpu"
    out_f, in_f = w_int8.shape
    w = w_int8.to(dev).float() * w_scale.to(dev)
    n_groups = in_f // group_size
    h = build_hadamard(group_size).to(dev)
    grouped = w.reshape(out_f, n_groups, group_size)
    out = torch.matmul(grouped, h.T).reshape(out_f, in_f)
    out = out.to(torch.bfloat16)
    if dev != "cpu":
        out = out.cpu()
    return out


def main():
    t0 = time.time()
    merged = {}

    # feature extractor from TEXT_ENC — KEEP source keys the ops map expects
    # (text_embedding_projection.{video,audio}_aggregate_embed.*)
    with safetensors.safe_open(TEXT_ENC, framework="pt") as f:
        for k in f.keys():
            if k.startswith("text_embedding_projection.video_aggregate_embed.") or \
               k.startswith("text_embedding_projection.audio_aggregate_embed."):
                merged[k] = f.get_tensor(k).to(torch.bfloat16).contiguous()
    print(f"[sidecar] feature_extractor {len(merged)} tensors ({time.time()-t0:.0f}s)", flush=True)

    # connectors from int8 source (convrot dequant) — KEEP full source keys the
    # ops map expects (model.diffusion_model.{video,audio}_embeddings_connector.*)
    with safetensors.safe_open(INT8, framework="pt") as f:
        keys = list(f.keys())
        n = 0
        for k in keys:
            if not k.startswith(PREFIX + "video_embeddings_connector.") and \
               not k.startswith(PREFIX + "audio_embeddings_connector."):
                continue
            if k.endswith(".weight_scale") or k.endswith(".comfy_quant"):
                continue
            if k.endswith(".weight"):
                sk = k + "_scale"
                ck = k.replace(".weight", ".comfy_quant")
                if sk in keys and ck in keys:
                    w = f.get_tensor(k)
                    ws = f.get_tensor(sk)
                    cq = f.get_tensor(ck)
                    try:
                        conf = json.loads(bytes(cq.tolist()).decode())
                        gs = conf.get("convrot_groupsize", 256)
                        merged[k] = dequant_convrot(w, ws, gs)
                    except Exception:
                        merged[k] = w.to(torch.bfloat16).contiguous()
                    n += 1
                    continue
            merged[k] = f.get_tensor(k).to(torch.bfloat16).contiguous()
            n += 1
    print(f"[sidecar] connectors {n} tensors ({time.time()-t0:.0f}s)", flush=True)

    # embed the int8 checkpoint's config metadata (needed to build the processor)
    from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
    meta = SafetensorsModelStateDictLoader().metadata(INT8)
    md = {k: (json.dumps(v, default=str) if not isinstance(v, str) else v)
          for k, v in meta.items()}
    print(f"[sidecar] embedding metadata keys: {list(meta.keys())}", flush=True)

    print(f"[sidecar] saving {len(merged)} tensors -> {DST}", flush=True)
    save_file(merged, DST, metadata=md)
    print(f"[sidecar] DONE {os.path.getsize(DST)/1e6:.1f} MB in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
