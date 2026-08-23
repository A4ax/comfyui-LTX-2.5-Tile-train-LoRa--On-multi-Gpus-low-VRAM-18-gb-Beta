"""N-GPU model-parallel int4 sharing via a block-sharded autograd pipeline.

Each rank owns a contiguous slice of the 48 transformer blocks (plus rank 0
owns the input/output aux). Forward passes intermediate hidden states down the
rank chain via a blocking `dist.send/recv`; gradients flow back up via custom
`autograd.Function` send/recv pairs, so LoRA grads update on the rank that owns
each block. Per-card VRAM = 10.1GB / N (vs 10.1GB on every card with DDP).

Pipeline layout (data path):
    rank0: patchify/prepare -> blocks[0:k0] -> SEND -> ... -> rank_last blocks -> RECV back -> output
Gradient path is the mirror reverse.

For N=2 (rank0 output, rank1 mid):
    forward : r0 blocks0..k -> send x -> r1 recv, blocks k..48 -> send y -> r0 recv, output
    backward: r0 ds/dy -> (recv.autograd sends to r1) -> r1 backward, send dBlock0 -> r0
"""
from __future__ import annotations

import os
import sys
import time

import engine_env  # noqa: E402
engine_env.setup_paths()
from msvc_env import apply_msvc_env  # noqa: E402

apply_msvc_env()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn as nn  # noqa: E402

from ltx_core.loader.helpers import create_meta_model  # noqa: E402
from ltx_core.model.transformer.model_configurator import LTXModelConfigurator  # noqa: E402
from optimum.quanto import freeze, quantize, qint2, qint4  # noqa: E402

# Quantization bit-width used when (re)quantizing a skeleton before loading a
# merged cache. int4 = 4-bit (tinygemm), int2 = 2-bit (plain QBits, half the
# dequant cost in backward). Select via LTX_QUANT_BITS env (2 or 4).
QUANT_BITS = int(os.environ.get("LTX_QUANT_BITS", "4"))
QUANTIZER = {2: qint2, 4: qint4}.get(QUANT_BITS, qint4)

import bitsandbytes as bnb  # noqa: E402
from bitsandbytes.functional import QuantState  # noqa: E402

CACHE_DIR = engine_env.CACHE_DIR
BNB_CACHE_DIR = os.path.join(engine_env.MODELS_ROOT, "diffusion_models", "ltx-bnb-nf4-cache")
PLAIN_MODULES = {"patchify_proj", "audio_patchify_proj", "proj_out", "audio_proj_out"}


def _linear_io(lin):
    """Return (out_features, in_features) of a linear-ish module, working for
    nn.Linear, quanto QLinear, and bnb Linear4bit (whose .weight is a packed
    Params4bit, so .weight.shape is NOT (out, in))."""
    if hasattr(lin, "in_features") and hasattr(lin, "out_features"):
        return lin.out_features, lin.in_features
    w = lin.weight
    return w.shape[0], w.shape[1]


def shard_ranges(n_blocks: int, world: int) -> list[tuple[int, int]]:
    """Contiguous [start,end) block ranges, one per rank, covering all blocks."""
    base = n_blocks // world
    rem = n_blocks % world
    out, acc = [], 0
    for r in range(world):
        size = base + (1 if r < rem else 0)
        out.append((acc, acc + size)); acc += size
    return out


def build_allocation(n_blocks: int, counts: list[int]) -> list[tuple[int, int]]:
    """Build per-rank [start,end) block ranges from explicit per-rank block counts.
    `counts[r]` = number of blocks GPU r holds. Must sum to n_blocks.
    Example: build_allocation(48, [16, 32]) -> [(0,16),(16,48)]."""
    if sum(counts) != n_blocks:
        raise ValueError(f"block counts {counts} sum to {sum(counts)} != n_blocks {n_blocks}")
    out, acc = [], 0
    for c in counts:
        out.append((acc, acc + c)); acc += c
    return out


_PLAIN_SD: dict | None = None


def _load_plain(module, name, device, t0):
    global _PLAIN_SD
    if _PLAIN_SD is None:
        _PLAIN_SD = torch.load(os.path.join(CACHE_DIR, "plain.pt"), map_location="cpu", weights_only=False)
    module.to_empty(device=device)
    module.to(torch.bfloat16)
    module.load_state_dict(_PLAIN_SD.get(name, {}), strict=False, assign=True)
    torch.cuda.synchronize(device)
    print(f"[shard] aux {name} (PLAIN) ({time.time()-t0:.0f}s)", flush=True)


def _load_one(module, path, device, label, t0):
    module.to_empty(device=device)
    module.to(torch.bfloat16)
    quantize(module, weights=QUANTIZER)
    freeze(module)
    sd = torch.load(path, map_location=str(device), weights_only=False)
    module.load_state_dict(sd, strict=False, assign=True)
    del sd
    torch.cuda.synchronize(device)
    print(f"[shard] {label} on {device} ({time.time()-t0:.0f}s)", flush=True)


def _load_shard_from_file(model, pt_path: str, device, start, end, t0) -> None:
    """Load the model directly from a self-contained merged int4 file.
    Supports .safetensors (fast, mmap, per-rank slicing) and .pt (torch.load).
    Aux (incl. plain bf16) loaded on every rank; only owned blocks [start,end)
    are materialized from the file's transformer_blocks.N.* keys."""
    # single-file int4: safetensors loads DIRECTLY into GPU (mmap, no CPU materialization);
    # .pt loads into host RAM then pushes to GPU.
    if pt_path.endswith(".safetensors"):
        _load_shard_from_safetensors(model, pt_path, device, start, end, t0)
        return
    else:
        sd_all = torch.load(pt_path, map_location="cpu", weights_only=False)
        print(f"[shard] loaded single file {os.path.getsize(pt_path)/1e9:.1f}GB ({len(sd_all)} tensors) ({time.time()-t0:.0f}s)", flush=True)
        aux_names = [n for n, _ in model.named_children() if n != "transformer_blocks"]
        for name in aux_names:
            mod = getattr(model, name)
            prefix = f"{name}."
            sub = {k[len(prefix):]: v for k, v in sd_all.items() if k.startswith(prefix)}
            if name in PLAIN_MODULES:
                mod.to_empty(device=device)
                mod.to(torch.bfloat16)
                mod.load_state_dict(sub, strict=False, assign=False)
                torch.cuda.synchronize(device)
                print(f"[shard] {name} (plain bf16) ({time.time()-t0:.0f}s)", flush=True)
            else:
                _quantize_and_load(mod, sub, device, f"aux {name}", t0)
        bare = {k: v.to(device) for k, v in sd_all.items() if "." not in k}
        if bare:
            model.load_state_dict(bare, strict=False, assign=True)
            print(f"[shard] bare params ({time.time()-t0:.0f}s)", flush=True)
        for bi in range(start, end):
            block = model.transformer_blocks[bi]
            prefix = f"transformer_blocks.{bi}."
            sub = {k[len(prefix):]: v for k, v in sd_all.items() if k.startswith(prefix)}
            _quantize_and_load(block, sub, device, f"block {bi}", t0)
        # free the 10GB host dict WITHOUT heavy gc.collect (slow on ~10k tensor objects)
        del sd_all
        torch.cuda.empty_cache()


def _load_shard_from_safetensors(model, pt_path, device, start, end, t0) -> None:
    """Load from a .safetensors int4 file DIRECTLY INTO GPU via safe_open (mmap).
    Tensors are read with safe_open(device=...) so no CPU materialization, and dtypes
    are preserved (uint8 packed `_data._data` stays uint8; scale/shift stay bf16) so
    quanto's _load_from_state_dict reconstructs WeightQBitsTensor correctly."""
    import safetensors
    dev = str(device)
    with safetensors.safe_open(pt_path, framework="pt", device=dev) as f:
        file_keys = set(f.keys())
        print(f"[shard] open {os.path.getsize(pt_path)/1e9:.1f}GB ({len(file_keys)} tensors) ({time.time()-t0:.0f}s)", flush=True)

        def get(key):
            # safe_open was opened with device=... so get_tensor lands on the GPU
            return f.get_tensor(key)

        aux_names = [n for n, _ in model.named_children() if n != "transformer_blocks"]
        for name in aux_names:
            mod = getattr(model, name)
            prefix = f"{name}."
            sub = {k[len(prefix):]: get(k) for k in file_keys if k.startswith(prefix)}
            if name in PLAIN_MODULES:
                mod.to_empty(device=device)
                mod.to(torch.bfloat16)
                mod.load_state_dict(sub, strict=False, assign=False)
                torch.cuda.synchronize(device)
                print(f"[shard] {name} (plain bf16) ({time.time()-t0:.0f}s)", flush=True)
            else:
                _quantize_and_load(mod, sub, device, f"aux {name}", t0)

        bare = {k: get(k) for k in file_keys if "." not in k}
        if bare:
            model.load_state_dict(bare, strict=False, assign=True)
            print(f"[shard] bare params ({time.time()-t0:.0f}s)", flush=True)

        for bi in range(start, end):
            block = model.transformer_blocks[bi]
            prefix = f"transformer_blocks.{bi}."
            sub = {k[len(prefix):]: get(k) for k in file_keys if k.startswith(prefix)}
            _quantize_and_load(block, sub, device, f"block {bi}", t0)


def _quantize_and_load(module, sd, device, label, t0) -> None:
    """to_empty + quantize(int4)+freeze then load a (nested quanto-keyed) sub-dict."""
    module.to_empty(device=device)
    module.to(torch.bfloat16)
    quantize(module, weights=QUANTIZER)
    freeze(module)
    module.load_state_dict(sd, strict=False, assign=True)
    del sd
    torch.cuda.synchronize(device)
    print(f"[shard] {label} on {device} ({time.time()-t0:.0f}s)", flush=True)


def _fill_bnb_block(blk, sd, device) -> None:
    """Reconstruct a bnb-NF4 block from a flat sub-dict (per-block `.pt` OR a slice
    of the single-file keys), rebuilding each Linear4bit via QuantState.from_dict
    (HF-style). GPU-only."""
    quant_paths = sorted({k[: -len(".weight.quant_map")] for k in sd if k.endswith(".weight.quant_map")})
    if not quant_paths:
        blk.load_state_dict(sd, strict=False, assign=True)
        return
    plain = {k: v for k, v in sd.items() if not k.startswith(tuple(f"{q}." for q in quant_paths))}
    blk.load_state_dict(plain, strict=False, assign=True)
    for qp in quant_paths:
        parts = qp.split(".")
        parent = blk
        for p in parts[:-1]:
            parent = getattr(parent, p)
        leaf = parts[-1]
        w_data = sd[f"{qp}.weight"]
        qs = QuantState.from_dict(
            {k: v for k, v in sd.items() if k.startswith(f"{qp}.weight.")}, device)
        out_f, in_f = qs.shape
        has_bias = f"{qp}.bias" in sd
        pw = bnb.nn.Params4bit(data=w_data, quant_state=qs, bnb_quantized=True,
                               blocksize=qs.blocksize, quant_type=qs.quant_type,
                               quant_storage=torch.uint8)
        lin = bnb.nn.Linear4bit(in_f, out_f, bias=has_bias,
                                compute_dtype=torch.bfloat16, quant_type=qs.quant_type,
                                compress_statistics=True)
        lin.weight = pw
        if has_bias:
            lin.bias.data = sd[f"{qp}.bias"]
        lin = lin.to(device)
        if isinstance(parent, nn.ModuleList) or isinstance(parent, nn.Sequential):
            parent[int(leaf)] = lin
        else:
            setattr(parent, leaf, lin)


def _load_bnb_block(model, bi: int, path: str, device) -> None:
    blk = model.transformer_blocks[bi]
    sd = torch.load(path, map_location="cpu", weights_only=False)
    _fill_bnb_block(blk, sd, device)


def _load_bnb_from_file(model, pt_path, device, start, end, t0) -> None:
    """Load aux + bare + owned blocks from a single self-contained bnb-NF4
    .safetensors directly into GPU (safe_open device=..., no CPU materialization).
    Skeleton config is read from the file's metadata header."""
    import safetensors
    import json as _json
    dev = str(device)
    with safetensors.safe_open(pt_path, framework="pt", device=dev) as f:
        file_keys = set(f.keys())
        print(f"[bnb] open {os.path.getsize(pt_path)/1e9:.1f}GB ({len(file_keys)} tensors) ({time.time()-t0:.0f}s)", flush=True)

        def get(key):
            return f.get_tensor(key)

        aux_names = [n for n, _ in model.named_children() if n != "transformer_blocks"]
        for name in aux_names:
            mod = getattr(model, name)
            prefix = f"{name}."
            sub = {k[len(prefix):]: get(k) for k in file_keys if k.startswith(prefix)}
            mod.to_empty(device=device)
            mod.to(torch.bfloat16)
            mod.load_state_dict(sub, strict=False, assign=True)
            torch.cuda.synchronize(device)
            print(f"[bnb] aux {name} bf16 ({time.time()-t0:.0f}s)", flush=True)

        bare = {k: get(k) for k in file_keys if "." not in k}
        if bare:
            model.load_state_dict(bare, strict=False, assign=True)
            print(f"[bnb] bare params ({time.time()-t0:.0f}s)", flush=True)

        for bi in range(start, end):
            prefix = f"transformer_blocks.{bi}."
            sub = {k[len(prefix):]: get(k) for k in file_keys if k.startswith(prefix)}
            _fill_bnb_block(model.transformer_blocks[bi], sub, device)
            print(f"[bnb] block {bi} nf4 ({time.time()-t0:.0f}s)", flush=True)


def load_bnb_shard(device, rank: int, world: int, counts: list[int] | None = None,
                   bnb_dir: str = BNB_CACHE_DIR, bnb_pt: str | None = None) -> torch.nn.Module:
    """Build the LTXModel with bnb-NF4 transformer blocks (fused 4-bit kernels ->
    fast backward) and bf16 aux. Blocks are sharded across ranks; aux is small
    and duplicated on every rank. Loads straight to GPU (never CPU).

    `bnb_pt`: optional path to a SELF-CONTAINED bnb-NF4 .safetensors (hosted).
    When provided, reads directly from that single file (per-rank block slicing,
    skeleton config from the file metadata header); otherwise reads the per-file
    `bnb_dir` cache."""
    device = torch.device(device)
    t0 = time.time()
    if bnb_pt:
        import safetensors
        import json as _json
        with safetensors.safe_open(bnb_pt, framework="pt") as f:
            meta = _json.loads(f.metadata().get("int4_config", "{}"))
        model = create_meta_model(LTXModelConfigurator, meta).to(dtype=torch.bfloat16)
        print(f"[bnb] rank{rank} skeleton {len(model.transformer_blocks)} blocks (from {os.path.basename(bnb_pt)}) ({time.time()-t0:.0f}s)", flush=True)
    else:
        with open(os.path.join(bnb_dir, "config.json"), encoding="utf-8") as fh:
            import json
            meta = json.load(fh).get("int4_config", {})
        model = create_meta_model(LTXModelConfigurator, meta).to(dtype=torch.bfloat16)
        print(f"[bnb] rank{rank} skeleton {len(model.transformer_blocks)} blocks ({time.time()-t0:.0f}s)", flush=True)

    n_blocks = len(model.transformer_blocks)
    if counts is not None:
        if len(counts) != world:
            raise ValueError(f"counts {counts} length != world {world}")
        ranges = build_allocation(n_blocks, counts)
    else:
        ranges = shard_ranges(n_blocks, world)
    start, end = ranges[rank]

    if bnb_pt:
        _load_bnb_from_file(model, bnb_pt, device, start, end, t0)
    else:
        # aux + bare (bf16) on EVERY rank
        aux = sorted(x for x in os.listdir(bnb_dir) if x.startswith("aux_") and x.endswith(".pt"))
        for fn in aux:
            name = fn[len("aux_"):-len(".pt")]
            mod = getattr(model, name)
            mod.to_empty(device=device)
            mod.to(torch.bfloat16)
            sd = torch.load(os.path.join(bnb_dir, fn), map_location=str(device), weights_only=False)
            mod.load_state_dict(sd, strict=False, assign=True)
            del sd
            torch.cuda.synchronize(device)
            print(f"[bnb] aux {name} bf16 ({time.time()-t0:.0f}s)", flush=True)
        bp = os.path.join(bnb_dir, "bare.pt")
        if os.path.exists(bp):
            bare = torch.load(bp, map_location=str(device), weights_only=False)
            model.load_state_dict(bare, strict=False, assign=True)
            del bare
            print(f"[bnb] bare params ({time.time()-t0:.0f}s)", flush=True)
        # owned blocks -> bnb NF4
        for bi in range(start, end):
            _load_bnb_block(model, bi, os.path.join(bnb_dir, f"block_{bi:02d}.pt"), device)
            print(f"[bnb] block {bi} nf4 ({time.time()-t0:.0f}s)", flush=True)

    ncpu = [nm for nm, p in model.named_parameters() if p.device.type == "cpu"]
    if ncpu:
        for nm, p in list(model.named_parameters()):
            if p.device.type == "cpu":
                p.data = p.data.to(device)
        ncpu = [nm for nm, p in model.named_parameters() if p.device.type == "cpu"]
        if ncpu:
            raise RuntimeError(f"[bnb] {len(ncpu)} params still on cpu: {ncpu[:5]}")
    torch.cuda.synchronize(device)
    print(f"[bnb] rank{rank} block shard[{start}:{end}] + aux, vram={torch.cuda.memory_allocated(device)/1e9:.1f}GB ({time.time()-t0:.0f}s)", flush=True)
    return model


def load_int4_shard(device: str | torch.device, rank: int, world: int,
                    counts: list[int] | None = None,
                    pt_path: str | None = None) -> torch.nn.Module:
    """Build the LTXModel with aux loaded on EVERY rank but only this rank's block
    shard materialized. Aux (~0.5GB) is duplicated cheaply; the ~10GB of block
    weights are shared, so per-rank VRAM â‰ˆ (blocks 10.1GB / N) + aux. Blocks not
    owned stay meta (zero VRAM).

    `counts`: optional explicit per-rank block counts (e.g. [16,32] for world=2).
    When None, blocks are split evenly via shard_ranges.

    `pt_path`: optional path to a SELF-CONTAINED merged int4 file (e.g. the
    HuggingFace-hosted ltx-...-int4-main-v2.pt). When provided, the loader reads
    directly from that single file instead of the 63-file cache dir â€” no cache
    needed, so users just download one file and train."""
    device = torch.device(device)
    t0 = time.time()
    with open(os.path.join(CACHE_DIR, "config.json"), encoding="utf-8") as fh:
        import json
        meta = json.load(fh).get("int4_config", {})
    model = create_meta_model(LTXModelConfigurator, meta).to(dtype=torch.bfloat16)
    print(f"[shard] rank{rank} skeleton {len(model.transformer_blocks)} blocks ({time.time()-t0:.0f}s)", flush=True)

    n_blocks = len(model.transformer_blocks)
    if counts is not None:
        if len(counts) != world:
            raise ValueError(f"counts {counts} length != world {world}")
        ranges = build_allocation(n_blocks, counts)
    else:
        ranges = shard_ranges(n_blocks, world)
    start, end = ranges[rank]

    if pt_path:
        _load_shard_from_file(model, pt_path, device, start, end, t0)
    else:
        # aux + bare on EVERY rank (small; cheap to duplicate for shared input/output use)
        aux = sorted(x for x in os.listdir(CACHE_DIR) if x.startswith("aux_") and x.endswith(".pt"))
        for fn in aux:
            name = fn[len("aux_"):-len(".pt")]
            mod = getattr(model, name)
            if name in PLAIN_MODULES:
                _load_plain(mod, name, device, t0)
            else:
                _load_one(mod, os.path.join(CACHE_DIR, fn), device, f"aux {name}", t0)
        bp = os.path.join(CACHE_DIR, "bare.pt")
        if os.path.exists(bp):
            bare = torch.load(bp, map_location=str(device), weights_only=False)
            model.load_state_dict(bare, strict=False, assign=True)
            del bare
            print(f"[shard] bare params ({time.time()-t0:.0f}s)", flush=True)

        # owned blocks only
        for bi in range(start, end):
            _load_one(model.transformer_blocks[bi], os.path.join(CACHE_DIR, f"block_{bi:02d}.pt"),
                      device, f"block {bi}", t0)

    # move any leftover CPU params (plain modules loaded from the sidecar) to `device`
    if pt_path is None:
        ncpu = [nm for nm, p in model.named_parameters() if p.device.type == "cpu"]
        if ncpu:
            for nm, p in list(model.named_parameters()):
                if p.device.type == "cpu":
                    p.data = p.data.to(device)
            ncpu = [nm for nm, p in model.named_parameters() if p.device.type == "cpu"]
            if ncpu:
                raise RuntimeError(f"[shard] {len(ncpu)} params still on cpu: {ncpu[:5]}")
            print(f"[shard] moved stray cpu params -> {device} ({time.time()-t0:.0f}s)", flush=True)
    torch.cuda.synchronize(device)
    print(f"[shard] rank{rank} block shard[{start}:{end}] + aux, vram={torch.cuda.memory_allocated(device)/1e9:.1f}GB ({time.time()-t0:.0f}s)", flush=True)
    return model


# ---------------------------------------------------------------------------
# autograd pipeline primitives (blocking gloo send/recv with grad threading)
# ---------------------------------------------------------------------------

class _AutogradRecv(torch.autograd.Function):
    """Receives a tensor from src. backward sends this gradient back to src."""

    @staticmethod
    def forward(ctx, src, shape, dtype, device):
        ctx.src = src
        t = torch.empty(shape, dtype=dtype, device=device)
        dist.recv(t, src=src)
        return t

    @staticmethod
    def backward(ctx, grad_output):
        if dist.get_world_size() > 1:
            dist.send(grad_output.detach().contiguous(), dst=ctx.src)
        return None, None, None, None


class _AutogradSend(torch.autograd.Function):
    """Sends `t` to dst in forward. backward receives the gradient from dst."""

    @staticmethod
    def forward(ctx, t, dst):
        ctx.dst = dst
        ctx.shape = tuple(t.shape)
        ctx.dtype = t.dtype
        ctx.device = t.device
        if dist.get_world_size() > 1:
            dist.send(t.detach().contiguous(), dst=dst)
        return t

    @staticmethod
    def backward(ctx, grad_output):
        if dist.get_world_size() > 1:
            g = torch.empty(ctx.shape, dtype=ctx.dtype, device=ctx.device)
            dist.recv(g, src=ctx.dst)
            return g, None
        return grad_output, None


def pipe_send(t: torch.Tensor, dst: int) -> torch.Tensor:
    return _AutogradSend.apply(t, dst)


def pipe_recv(src: int, shape, dtype, device) -> torch.Tensor:
    return _AutogradRecv.apply(src, tuple(shape), dtype, device)


# ---------------------------------------------------------------------------
# lightweight LoRA (PEFT can't wrap a model whose unowned blocks are meta)
# ---------------------------------------------------------------------------

class _LoRA(nn.Module):
    """Wrap a frozen nn.Linear with a low-rank update out = base(x) + (x@A^T)@B^T * scale."""

    def __init__(self, base: nn.Module, r: int = 16, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        out0, in0 = _linear_io(base)
        dev = next(base.parameters()).device
        dt = torch.bfloat16
        self.lora_A = nn.Parameter(torch.zeros(r, in0, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(out0, r, device=dev, dtype=dt))
        with torch.no_grad():
            self.lora_A.normal_(std=0.02)
        self.scale = alpha / r

    def forward(self, x):
        # Standard LoRA: base weights are frozen (requires_grad_(False)) so they
        # don't update, but gradient flows THROUGH the base to reach earlier
        # layers. (The old .detach() here severed that path -> divergent training;
        # with bnb's fused backward the base pass is fast enough to keep intact.)
        base_out = self.base(x)
        return base_out + (x @ self.lora_A.t()) @ self.lora_B.t() * self.scale

    @property
    def weight(self):
        # mirror nn.Linear interface in case anything reads .weight
        return self.base.weight


def _looks_like_proj(lin) -> bool:
    # In bf16 the linears are nn.Linear; after int4 they are quanto QLinear (exposes .weight).
    return hasattr(lin, "weight") and hasattr(lin, "forward")


def _add_lora_one_block(block: nn.Module, r: int = 16, alpha: float = 16.0) -> int:
    n = 0
    for attn_name in ["attn1", "attn2", "audio_attn1", "audio_attn2",
                      "audio_to_video_attn", "video_to_audio_attn"]:
        attn = getattr(block, attn_name, None)
        if attn is None:
            continue
        for lin_name in ["to_q", "to_k", "to_v", "to_gate_logits"]:
            lin = getattr(attn, lin_name, None)
            if _looks_like_proj(lin):
                setattr(attn, lin_name, _LoRA(lin, r, alpha))
                _o, _i = _linear_io(lin)
                n += r * (_o + _i)
        out = getattr(attn, "to_out", None)
        if isinstance(out, nn.Sequential) and _looks_like_proj(out[0]):
            out[0] = _LoRA(out[0], r, alpha)
            _o, _i = _linear_io(out[0].base)
            n += r * (_o + _i)
    return n


def add_lora_to_block(model: nn.Module, r: int = 16, alpha: float = 16.0,
                      start: int = 0, end: int | None = None) -> int:
    """Wrap attention projection linears in Lightweight LoRA for owned (materialized)
    blocks only (blocks in [start,end)). Handles both nn.Linear and quanto QLinear.
    Returns number of LoRA params added."""
    end = len(model.transformer_blocks) if end is None else end
    n = 0
    for bi in range(start, end):
        blk = model.transformer_blocks[bi]
        if _block_is_meta(blk):
            continue
        n += _add_lora_one_block(blk, r, alpha)
    return n


def _block_is_meta(block: nn.Module) -> bool:
    p = next(block.parameters(), None)
    return p is not None and p.device.type == "meta"


# Attention modules we wrap in LoRA, in the peft/ComfyUI key path.
_ATTN_NAMES = ["attn1", "attn2", "audio_attn1", "audio_attn2",
               "audio_to_video_attn", "video_to_audio_attn"]
_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_gate_logits", "to_out.0"]


def collect_lora_state_dict(model: nn.Module, start: int = 0, end: int | None = None) -> dict:
    """Collect this rank's LoRA params into peft/ComfyUI-style keys:
       diffusion_model.transformer_blocks.<N>.<attn>.<target>.lora_{A,B}.weight
    Only owned (materialized) blocks in [start,end) are included."""
    end = len(model.transformer_blocks) if end is None else end
    sd = {}
    for bi in range(start, end):
        blk = model.transformer_blocks[bi]
        if _block_is_meta(blk):
            continue
        for attn_name in _ATTN_NAMES:
            attn = getattr(blk, attn_name, None)
            if attn is None:
                continue
            for target in _LORA_TARGETS:
                lin = _resolve_lora_linear(attn, target)
                if isinstance(lin, _LoRA):
                    prefix = f"diffusion_model.transformer_blocks.{bi}.{attn_name}.{target}."
                    sd[prefix + "lora_A.weight"] = lin.lora_A.detach().float().cpu()
                    sd[prefix + "lora_B.weight"] = lin.lora_B.detach().float().cpu()
    return sd


def _resolve_lora_linear(attn, target: str):
    if target == "to_out.0":
        out = getattr(attn, "to_out", None)
        return out[0] if isinstance(out, nn.Sequential) and out else None
    return getattr(attn, target, None)
