"""Build a bnb-NF4 int4 cache from int8-convrot, saving PER-BLOCK modules.

Each transformer block's nn.Linear (attention + FFN) is converted to a
bitsandbytes `Linear4bit` (NF4) with fused 4-bit kernels -> fast backward.
Blocks are saved via torch.save(module) so the Params4bit weight + its
QuantState pickle losslessly. Aux + plain + bare stay bf16 (small, and bf16
aux is higher quality than int4). GPU-only.

Usage:
  python build_bnb_nf4.py                # all 48 blocks
  python build_bnb_nf4.py --maxblocks 1  # block 0 only (validation)
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import os  # noqa: E402
import engine_env  # noqa: E402
engine_env.setup_paths()
from msvc_env import apply_msvc_env  # noqa: E402
apply_msvc_env()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import argparse  # noqa: E402
import json  # noqa: E402
import gc  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import bitsandbytes as bnb  # noqa: E402

import safetensors  # noqa: E402
from ltx_core.loader.helpers import create_meta_model  # noqa: E402
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader  # noqa: E402
from ltx_core.model.transformer.model_configurator import LTXModelConfigurator  # noqa: E402

SRC = os.path.join(engine_env.MODELS_ROOT, "diffusion_models", "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors")
CACHE_DIR = os.path.join(engine_env.MODELS_ROOT, "diffusion_models", "ltx-bnb-nf4-cache")
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
    del w
    return out


def gather_sd(f, prefix, file_keys):
    sd = {}
    for k in file_keys:
        if not k.startswith(prefix):
            continue
        rel = k[len(prefix):]
        if rel.endswith(".weight_scale") or rel.endswith(".comfy_quant"):
            continue
        if rel.endswith(".weight"):
            sk = k + "_scale"
            ck = k.replace(".weight", ".comfy_quant")
            if sk in file_keys and ck in file_keys:
                w = f.get_tensor(k)
                ws = f.get_tensor(sk)
                cq = f.get_tensor(ck)
                try:
                    conf = json.loads(bytes(cq.tolist()).decode())
                    gs = conf.get("convrot_groupsize", 256)
                    sd[rel] = dequant_convrot(w, ws, gs)
                except Exception:
                    sd[rel] = w.to(torch.bfloat16)
                continue
        sd[rel] = f.get_tensor(k).to(torch.bfloat16)
    return sd


def bootstrap_to_cuda(module):
    module.to_empty(device=_device)
    module.to(torch.bfloat16)


def linear_to_bnb(lin: torch.nn.Linear) -> bnb.nn.Linear4bit:
    """Copy an nn.Linear's bf16 weights into a bnb Linear4bit (NF4) and quantize."""
    in_f = lin.weight.shape[1]
    out_f = lin.weight.shape[0]
    has_bias = lin.bias is not None
    w = lin.weight.data.detach().float().cpu()  # bf16 build-time only, per-layer
    b = lin.bias.data.detach().float().cpu() if has_bias else None
    new = bnb.nn.Linear4bit(in_f, out_f, bias=has_bias,
                            compute_dtype=torch.bfloat16, quant_type="nf4",
                            compress_statistics=True)
    new.weight.data = w
    if has_bias:
        new.bias.data = b
    new = new.to(_device)  # triggers NF4 quantization of weight
    return new


def quantize_block_bnb(block):
    """Convert every nn.Linear in the block to bnb Linear4bit, in place."""
    n = 0
    for name, mod in list(block.named_modules()):
        if not (hasattr(mod, "weight") and "Linear" in type(mod).__name__):
            continue
        parent = block
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        if isinstance(parent, torch.nn.ModuleList) or isinstance(parent, torch.nn.Sequential):
            parent[int(parts[-1])] = linear_to_bnb(mod)
        else:
            setattr(parent, parts[-1], linear_to_bnb(mod))
        n += 1
    return n


def verify_block_fwd(block):
    """Smoke-test the bnb FFNs (the speed-critical path)."""
    torch.manual_seed(0)
    x = torch.randn(1, 64, 4096, device=_device, dtype=torch.bfloat16)
    shapes = []
    with torch.no_grad():
        v = block.ff(x)
        shapes.append(tuple(v.shape))
        if hasattr(block, "audio_ff"):
            ax = torch.randn(1, 64, 2048, device=_device, dtype=torch.bfloat16)
            shapes.append(tuple(block.audio_ff(ax).shape))
    return shapes


def save_block(block, path):
    """Serialize a block's state_dict (flat tensors incl. bnb `weight.*` metadata)."""
    torch.save(block.state_dict(), path)


def rebuild_block(path, dtype=torch.bfloat16, device=_device):
    """Rebuild a bnb block from a saved state_dict straight to GPU, reconstructing
    each Linear4bit via QuantState.from_dict (HF-style)."""
    from bitsandbytes.functional import QuantState
    from ltx_core.loader.helpers import create_meta_model
    from ltx_core.model.transformer.model_configurator import LTXModelConfigurator
    from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader

    meta = SafetensorsModelStateDictLoader().metadata(SRC)
    m = create_meta_model(LTXModelConfigurator, meta).to(dtype=dtype)
    blk = m.transformer_blocks[0]
    sd = torch.load(path, map_location="cpu", weights_only=False)

    # find bnb quantized linear paths (have a sibling `<path>.weight.quant_map`)
    quant_paths = sorted({k[: -len(".weight.quant_map")]
                          for k in sd if k.endswith(".weight.quant_map")})

    plain = {k: v for k, v in sd.items() if not k.startswith(tuple(f"{q}." for q in quant_paths))}
    blk.load_state_dict(plain, strict=False, assign=True)

    for qp in quant_paths:
        parts = qp.split(".")
        parent = blk
        for p in parts[:-1]:
            parent = getattr(parent, p)
        leaf = parts[-1]
        w_data = sd[f"{qp}.weight"]
        qs_dict = {k: v for k, v in sd.items() if k.startswith(f"{qp}.weight.")}
        qs = QuantState.from_dict(qs_dict, device)
        out_f, in_f = qs.shape
        has_bias = f"{qp}.bias" in sd
        pw = bnb.nn.Params4bit(
            data=w_data, quant_state=qs, bnb_quantized=True,
            blocksize=qs.blocksize, quant_type=qs.quant_type,
            quant_storage=torch.uint8,
        )
        lin = bnb.nn.Linear4bit(in_f, out_f, bias=has_bias,
                                compute_dtype=dtype, quant_type=qs.quant_type,
                                compress_statistics=True)
        lin.weight = pw
        if has_bias:
            lin.bias.data = sd[f"{qp}.bias"]
        lin = lin.to(device)
        if isinstance(parent, torch.nn.ModuleList) or isinstance(parent, torch.nn.Sequential):
            parent[int(leaf)] = lin
        else:
            setattr(parent, leaf, lin)

    blk = blk.to(device)
    return blk


def save_aux_bare(cache):
    """Save all aux modules + bare params as bf16 state_dicts (self-contained)."""
    t0 = time.time()
    meta = SafetensorsModelStateDictLoader().metadata(SRC)
    model = create_meta_model(LTXModelConfigurator, meta).to(dtype=torch.bfloat16)
    with safetensors.safe_open(SRC, framework="pt") as f:
        file_keys = set(f.keys())
        for mod_name in [n for n, _ in model.named_children() if n != "transformer_blocks"]:
            module = getattr(model, mod_name)
            pfx = f"{PREFIX}{mod_name}."
            sub = gather_sd(f, pfx, file_keys)
            bootstrap_to_cuda(module)
            module.load_state_dict(sub, strict=False, assign=True)
            torch.save({k: v.detach().cpu().contiguous() for k, v in module.state_dict().items()},
                       os.path.join(cache, f"aux_{mod_name}.pt"))
            print(f"[BNB] aux {mod_name} bf16 saved ({time.time()-t0:.0f}s)", flush=True)
        bare = {}
        for bn in ["keyframes_abs_pos_embedding", "scale_shift_table", "audio_scale_shift_table"]:
            k = PREFIX + bn
            if k in file_keys:
                bare[bn] = f.get_tensor(k).to(torch.bfloat16).contiguous()
        if bare:
            torch.save(bare, os.path.join(cache, "bare.pt"))
            print(f"[BNB] bare bf16 saved {list(bare)} ({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(cache, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"int4_config": meta}, fh, default=str)
    print(f"[BNB] aux+bare DONE ({time.time()-t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxblocks", type=int, default=48)
    ap.add_argument("--dst", default=CACHE_DIR)
    ap.add_argument("--testload", action="store_true", help="load block_00.pt back and smoke-test")
    ap.add_argument("--auxonly", action="store_true", help="save only aux+bare bf16")
    args = ap.parse_args()
    cache = args.dst
    os.makedirs(cache, exist_ok=True)

    if args.auxonly:
        save_aux_bare(cache)
        return

    if args.testload:
        p0 = os.path.join(cache, "block_00.pt")
        blk = rebuild_block(p0)
        print("[BNB] rebuilt block loaded; running ff fwd...", flush=True)
        x = torch.randn(1, 64, 4096, device=_device, dtype=torch.bfloat16)
        with torch.no_grad():
            out = blk.ff(x)
        print(f"[BNB] ff fwd ok -> {tuple(out.shape)} PASS", flush=True)
        return

    t0 = time.time()
    print(f"[BNB] src {SRC} ({os.path.getsize(SRC)/1e9:.1f}GB) gpu={torch.cuda.get_device_name(0)}", flush=True)
    meta = SafetensorsModelStateDictLoader().metadata(SRC)
    model = create_meta_model(LTXModelConfigurator, meta).to(dtype=torch.bfloat16)
    n_blocks = min(len(model.transformer_blocks), args.maxblocks)
    print(f"[BNB] skeleton {n_blocks} blocks (of {len(model.transformer_blocks)})", flush=True)

    with safetensors.safe_open(SRC, framework="pt") as f:
        file_keys = set(f.keys())
        for bi in range(n_blocks):
            block = model.transformer_blocks[bi]
            pfx = f"{PREFIX}transformer_blocks.{bi}."
            block_sd = gather_sd(f, pfx, file_keys)
            bootstrap_to_cuda(block)
            block.load_state_dict(block_sd, strict=False, assign=True)
            del block_sd
            n_lin = quantize_block_bnb(block)
            shape = verify_block_fwd(block)
            path = os.path.join(cache, f"block_{bi:02d}.pt")
            save_block(block, path)
            gb = os.path.getsize(path) / 1e9
            model.transformer_blocks[bi] = None
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[BNB] block {bi} {n_lin} lin -> bnb-nf4 fwd{shape} saved {gb:.2f}GB ({time.time()-t0:.0f}s)", flush=True)

    with open(os.path.join(cache, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"int4_config": meta}, fh, default=str)
    tot = sum(os.path.getsize(os.path.join(cache, fn)) for fn in os.listdir(cache)
              if fn.endswith(".pt")) / 1e9
    print(f"[BNB] DONE in {time.time()-t0:.0f}s. cache {tot:.1f}GB", flush=True)


if __name__ == "__main__":
    main()
