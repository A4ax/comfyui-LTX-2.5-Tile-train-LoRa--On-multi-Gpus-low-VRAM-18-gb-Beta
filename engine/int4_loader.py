"""Load the int4 cache into a quanto LTXModel, BLOCK-BY-BLOCK, entirely on GPU.

Loads each module from the per-module cache files (built on 3 GPUs) directly
onto `device`, one block/aux at a time:

    block.to_empty(device) -> to(bf16) -> quantize(qint4)+freeze  (small)
    sd = torch.load(block_NN.pt, map_location=device)
    block.load_state_dict(sd, strict=False, assign=True)           (small)

Each block is ~0.23GB int4, so peak per-block is ~0.5GB; the 48 blocks
accumulate to ~11GB on `device`. Nothing is ever materialized on CPU.
The whole-model empty-skeleton + full state_dict path OOMs on a 12GB card,
so this per-module route is required.
"""
from __future__ import annotations

import os
import sys
import time

import engine_env  # noqa: E402
engine_env.setup_paths()
from msvc_env import apply_msvc_env  # noqa: E402

apply_msvc_env()  # quanto's C++ unpack ext (ninja + cl) JIT-builds

import torch  # noqa: E402

from ltx_core.loader.helpers import create_meta_model  # noqa: E402
from ltx_core.model.transformer.model_configurator import LTXModelConfigurator  # noqa: E402
from optimum.quanto import freeze, quantize, qint4  # noqa: E402

CACHE_DIR = engine_env.CACHE_DIR

# Modules the trainer's quantization excludes — MUST stay plain bf16 (never int4).
# These are input/output projections that need full precision.
PLAIN_MODULES = {"patchify_proj", "audio_patchify_proj", "proj_out", "audio_proj_out"}


_PLAIN_SD: dict | None = None


def _load_plain_from_src(module: torch.nn.Module, name: str, device: torch.device, t0: float) -> None:
    """Load a non-quantized (plain bf16) top-level module from the small plain.pt sidecar.
    Reading the 21.5GB int8 source from 2 ranks concurrently triggers native crashes;
    the sidecar (~3MB) is pre-extracted by extract_plain.py and safe for both ranks."""
    global _PLAIN_SD
    if _PLAIN_SD is None:
        _PLAIN_SD = torch.load(os.path.join(CACHE_DIR, "plain.pt"), map_location="cpu", weights_only=False)
    module.to_empty(device=device)
    module.to(torch.bfloat16)
    sd = _PLAIN_SD.get(name, {})
    module.load_state_dict(sd, strict=False, assign=True)
    del sd
    torch.cuda.synchronize(device)
    print(f"[int4] aux {name} (PLAIN bf16 sidecar) ({time.time()-t0:.0f}s)", flush=True)


def _load_one(module: torch.nn.Module, path: str, device: torch.device, label: str, t0: float) -> None:
    """Allocate + quantize + load ONE module on `device`. Fully on GPU."""
    module.to_empty(device=device)
    module.to(torch.bfloat16)
    quantize(module, weights=qint4)
    freeze(module)
    sd = torch.load(path, map_location=str(device), weights_only=False)
    module.load_state_dict(sd, strict=False, assign=True)
    del sd
    torch.cuda.synchronize(device)
    print(f"[int4] {label} on {device} ({time.time()-t0:.0f}s, vram={torch.cuda.memory_allocated(device)/1e9:.1f}GB)", flush=True)


def load_int4(device: str | torch.device = "cuda:0") -> torch.nn.Module:
    """Build the int4 LTXModel block-by-block entirely on `device` (GPU)."""
    if not torch.cuda.is_available():
        raise RuntimeError("load_int4 requires CUDA; building on CPU is disallowed.")
    device = torch.device(device)
    t0 = time.time()
    with open(os.path.join(CACHE_DIR, "config.json"), encoding="utf-8") as fh:
        import json
        meta = json.load(fh).get("int4_config", {})

    model = create_meta_model(LTXModelConfigurator, meta)
    model.to(dtype=torch.bfloat16)
    print(f"[int4] skeleton {type(model).__name__}, {len(model.transformer_blocks)} blocks ({time.time()-t0:.0f}s)", flush=True)

    # aux modules (each small, straight onto GPU)
    for fn in sorted(x for x in os.listdir(CACHE_DIR) if x.startswith("aux_") and x.endswith(".pt")):
        name = fn[len("aux_"):-len(".pt")]
        if name in PLAIN_MODULES:
            mod = getattr(model, name)
            _load_plain_from_src(mod, name, device, t0)
            continue
        mod = getattr(model, name)
        _load_one(mod, os.path.join(CACHE_DIR, fn), device, f"aux {name}", t0)

    # bare top-level params (small, load onto GPU directly)
    bp = os.path.join(CACHE_DIR, "bare.pt")
    if os.path.exists(bp):
        bare = torch.load(bp, map_location=str(device), weights_only=False)
        model.load_state_dict(bare, strict=False, assign=True)
        del bare
        print(f"[int4] bare params on {device} ({time.time()-t0:.0f}s)", flush=True)

    # transformer blocks, one at a time on GPU
    for bi in range(len(model.transformer_blocks)):
        block = model.transformer_blocks[bi]
        _load_one(block, os.path.join(CACHE_DIR, f"block_{bi:02d}.pt"), device, f"block {bi}", t0)
        if bi % 4 == 3:
            print(f"[int4]   ...{bi+1}/48 blocks on GPU ({time.time()-t0:.0f}s)", flush=True)

    uninit = [n for n, p in model.named_parameters() if p.numel() > 0 and p.device.type == "meta"]
    ncpu = [n for n, p in model.named_parameters() if p.device.type == "cpu"]
    print(f"[int4] uninitialized: {len(uninit)}, on_cpu: {len(ncpu)}", flush=True)
    if len(uninit) != 0:
        raise RuntimeError(f"int4 load left uninitialized params: {uninit}")
    if len(ncpu) != 0:
        print(f"[int4] moving {len(ncpu)} cpu params to {device}: {ncpu[:8]}", flush=True)
        for nm, p in list(model.named_parameters()):
            if p.device.type == "cpu":
                p.data = p.data.to(device)
        ncpu = [n for n, p in model.named_parameters() if p.device.type == "cpu"]
        if len(ncpu) != 0:
            raise RuntimeError(f"int4 load still left {len(ncpu)} params on cpu: {ncpu[:5]}")
        print(f"[int4] all params now on {device}", flush=True)
    b0 = model.transformer_blocks[0]
    print(f"[int4] block0.attn1.to_q.weight type: {type(b0.attn1.to_q.weight).__name__}", flush=True)
    print(f"[int4] DONE in {time.time()-t0:.0f}s, total GPU vram={torch.cuda.memory_allocated(device)/1e9:.1f}GB", flush=True)
    return model


if __name__ == "__main__":
    m = load_int4("cuda:0")
    print("[int4] PASS", flush=True)
