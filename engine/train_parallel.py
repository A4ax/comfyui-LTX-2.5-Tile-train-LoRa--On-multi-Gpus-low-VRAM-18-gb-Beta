"""Multi-step training on the N-GPU sharded int4 engine, with tiling, explicit
per-GPU block allocation, run-config JSON, telemetry, and checkpointing.

Driven by a run-config JSON (written by the ComfyUI node) so ComfyUI only passes
one file. Telemetry (loss / step time / steps-per-sec / ETA / VRAM) is appended
as JSON lines for the LogsOutputs node to tail.

Run (via multigpu_run, one process per GPU):
  engine_python -u multigpu_run.py --world N --devices 0,.. \\
      --config working/ddp_2gpu.yaml \\
      --script working/train_parallel.py -- --config <run.json>
"""
import argparse
import json
import os
import sys
import time
import types
from dataclasses import replace

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ENGINE_DIR)
from msvc_env import apply_msvc_env  # noqa: E402
apply_msvc_env()
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
device = torch.device(f"cuda:{rank}")
USE_NCCL = (sys.platform != "win32") and dist.is_nccl_available()
dist.init_process_group("nccl" if USE_NCCL else "gloo",
                        init_method="tcp://127.0.0.1:29570", rank=rank, world_size=world)
print(f"[TR] dist backend={'nccl' if USE_NCCL else 'gloo'}", flush=True)


def _comm(x, dtype=None):
    """Tensor for dist send/recv: CPU for gloo, GPU for NCCL."""
    t = x.detach().to(dtype) if dtype is not None else x.detach()
    return t.cpu() if not USE_NCCL else t


def _comm_empty(x, dtype=None):
    z = torch.empty_like(x)
    if dtype is not None:
        z = z.to(dtype)
    return z.cpu() if not USE_NCCL else z


def _comm_zeros(x, dtype):
    z = torch.zeros_like(x, dtype=dtype)
    return z.cpu() if not USE_NCCL else z


_PKG = os.environ.get("LTX_PACKAGES_DIR") or os.path.join(os.path.dirname(_ENGINE_DIR), "packages")
sys.path.insert(0, os.path.join(_PKG, "ltx-trainer", "src"))
sys.path.insert(0, os.path.join(_PKG, "ltx-core", "src"))

from ltx_core.components.patchifiers import VideoLatentPatchifier, AudioPatchifier, get_pixel_coords  # noqa: E402
from ltx_core.model.transformer.modality import Modality  # noqa: E402
from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape, AudioLatentShape  # noqa: E402
from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType  # noqa: E402

sys.path.insert(0, _ENGINE_DIR)
# Read the run config early so we can set the quantization bit-width (2 or 4)
# BEFORE int4_parallel is imported (it reads LTX_QUANT_BITS at import time).
_early_cfg = {}
import argparse as _argparse  # noqa: E402
_ap = _argparse.ArgumentParser(); _ap.add_argument("--config", default=None)
_ea, _ = _ap.parse_known_args()
if _ea.config:
    try:
        with open(_ea.config, encoding="utf-8") as _f:
            _early_cfg = json.load(_f)
    except Exception:
        _early_cfg = {}
_bits = int(_early_cfg.get("bits", 0))
if _bits not in (2, 4):
    _fn = str(_early_cfg.get("int4_pt") or "").lower()
    _bits = 2 if "int2" in _fn else 4
os.environ.setdefault("LTX_QUANT_BITS", str(_bits))
from int4_parallel import (  # noqa: E402
    load_int4_shard, load_bnb_shard, shard_ranges, build_allocation,
    add_lora_to_block, collect_lora_state_dict,
)
from tiling_feature import resolve_tiling, build_tiles, blend_tiles  # noqa: E402

W, H, F = 512, 512, 17
FPS = 24.0
SF = SpatioTemporalScaleFactors(time=8, height=32, width=32)
LF, LH, LW = (F - 1) // SF.time + 1, H // SF.height, W // SF.width
SEQ = LF * LH * LW
N_CTX = 256
DIM = 3840
CH = 128
DEFAULT_LR = 3e-4

DATASET_ROOT = None
NUM_SAMPLES = 0
AUDIO_CH = 8
AUDIO_FB = 16
TRAIN_AUDIO = False

# RAM cache of the whole dataset (video latents, condition embeds, audio latents),
# loaded ONCE at training start to eliminate per-step disk I/O.
_DS_CACHE = None

# GPU-side caches, built lazily. x0 (clean latent) is tiny -> cached on every rank.
# ctx (~8 MB/sample) is large -> cached on GPU only by the OWNING rank
# (idx % world == rank) and fetched from the owner via dist each step, so the big
# ctx cache is SHARED across GPUs instead of duplicated on both.
_GPU_CACHE_X = {}
_GPU_CACHE_CTX = {}

# Cached video pixel-coords (deterministic given the latent shape); avoids recomputing
# get_pixel_coords + a fresh patchifier every step.
_COORDS_CACHE = {}


def _video_coords(device_, dtype, lf, lh, lw, fps):
    key = (str(device_), str(dtype), lf, lh, lw, fps)
    c = _COORDS_CACHE.get(key)
    if c is None:
        p = VideoLatentPatchifier(1)
        shape = VideoLatentShape(batch=1, channels=CH, frames=lf, height=lh, width=lw)
        c = get_pixel_coords(p.get_patch_grid_bounds(shape, device=device_), SF, causal_fix=True).to(dtype)
        c[:, 0, ...] = c[:, 0, ...] / fps
        _COORDS_CACHE[key] = c.contiguous()
    return _COORDS_CACHE[key]


def _preload_dataset():
    """Load all dataset tensors into RAM once (keyed by sample idx)."""
    global _DS_CACHE
    _DS_CACHE = {}
    for idx in range(NUM_SAMPLES):
        entry = {}
        base = os.path.join(DATASET_ROOT, "latents", "scenes", f"img_{idx:03d}.pt")
        if os.path.exists(base):
            try:
                ld = torch.load(base, map_location="cpu", weights_only=True)
                entry["latents"] = ld["latents"] if isinstance(ld, dict) and "latents" in ld else ld
            except Exception:
                entry["latents"] = None
        cbase = os.path.join(DATASET_ROOT, "conditions", "scenes", f"img_{idx:03d}.pt")
        if os.path.exists(cbase):
            try:
                cd = torch.load(cbase, map_location="cpu", weights_only=True)
                entry["ctx"] = cd.get("video_prompt_embeds") if isinstance(cd, dict) else None
                entry["video_embeds"] = cd.get("video_embeds") if isinstance(cd, dict) else None
                entry["audio_emb"] = cd.get("audio_embeds") if isinstance(cd, dict) else None
            except Exception:
                entry["ctx"] = entry["video_embeds"] = entry["audio_emb"] = None
        abase = os.path.join(DATASET_ROOT, "audio_latents", "scenes", f"img_{idx:03d}.pt")
        if os.path.exists(abase):
            try:
                ad = torch.load(abase, map_location="cpu", weights_only=True)
                entry["audio_lat"] = ad["latents"] if isinstance(ad, dict) and "latents" in ad else ad
            except Exception:
                entry["audio_lat"] = None
        _DS_CACHE[idx] = entry
    n_audio = sum(1 for e in _DS_CACHE.values() if e.get("audio_lat") is not None)
    print(f"[TR] dataset preloaded into RAM: {len(_DS_CACHE)} samples ({n_audio} with audio)", flush=True)


def _apply_embeddings(idx, device_, dtype):
    """Load the CACHED connector-applied embeds from the condition file.
    The audio/video connectors are applied OFFLINE during preprocessing
    (precompute_audio_embeds.py), so training does NOT load the embeddings
    processor -> keeps VRAM at face-only levels. Returns (video_embeds,
    audio_embeds) or (None, None) when absent (audio branch skipped)."""
    e = (_DS_CACHE or {}).get(idx) if _DS_CACHE is not None else None
    if e is None:
        # fallback: read from disk
        cbase = os.path.join(DATASET_ROOT, "conditions", "scenes", f"img_{idx:03d}.pt")
        if not os.path.exists(cbase):
            return None, None
        cd = torch.load(cbase, map_location="cpu", weights_only=True)
        e = {"video_embeds": cd.get("video_embeds"), "audio_emb": cd.get("audio_embeds")}
    ve = e.get("video_embeds")
    ae = e.get("audio_emb")
    if ve is not None:
        ve = ve.unsqueeze(0).to(device_, dtype=dtype)
    ae_t = ae.unsqueeze(0).to(device_, dtype=dtype) if ae is not None else None
    return ve, ae_t


def _load_audio_latent(idx, device_, dtype):
    """Load the audio latent (8, T, 16) for sample idx."""
    e = (_DS_CACHE or {}).get(idx) if _DS_CACHE is not None else None
    if e is not None:
        lat = e.get("audio_lat")
        return lat.to(device_, dtype=dtype) if lat is not None else None
    base = os.path.join(DATASET_ROOT, "audio_latents", "scenes", f"img_{idx:03d}.pt")
    if not os.path.exists(base):
        return None
    ld = torch.load(base, map_location="cpu", weights_only=True)
    lat = ld["latents"] if isinstance(ld, dict) and "latents" in ld else ld
    return lat.to(device_, dtype=dtype)                               # (8,T,16)


def _audio_positions(T, device_, dtype, fps=FPS):
    """Audio positional coords [1,1,T,2] (n_pos_dims=1, time normalized by fps)."""
    p = torch.linspace(0, (T - 1) / fps, T, device=device_, dtype=dtype)
    return p.view(1, 1, T, 1).expand(1, 1, T, 2).contiguous()


def load_run_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_modality(device_, dtype, seed=0, lf=None, lh=None, lw=None, seq=None, fps=FPS):
    """Build a training sample from REAL dataset latents + conditions (flow matching).

    Returns (mod, target): mod.latent = noised latent x_t; target = velocity.
    """
    lf = lf or LF
    lh = lh or LH
    lw = lw or LW
    seq = seq or SEQ
    coords = _video_coords(device_, dtype, lf, lh, lw, fps)

    # ---- real data ----
    idx = (seed % NUM_SAMPLES) if NUM_SAMPLES > 0 else seed
    x0, ctx = _load_real_sample(idx, device_, dtype, seq)   # (1,seq,CH), (1,NCTX,CDIM)
    g = torch.Generator(device=device_).manual_seed(seed)   # GPU-only RNG (no CPU)
    noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=g)
    t = torch.rand(1, device=device_, dtype=dtype, generator=g)  # noise level 0..1
    x_t = (1.0 - t) * x0 + t * noise              # interpolate toward noise
    velocity = noise - x0                          # flow-matching target
    timesteps = torch.full((1, seq), t.item(), device=device_, dtype=dtype)
    sigma = torch.full((1,), t.item(), device=device_, dtype=dtype)
    mod = Modality(enabled=True, latent=x_t, sigma=sigma,
                   timesteps=timesteps, positions=coords.contiguous(), context=ctx, context_mask=None)
    return mod, velocity


def _load_real_sample(idx, device_, dtype, seq):
    """Load a real latent + condition for sample `idx`.

    x0 (tiny clean latent) is cached on every rank's GPU. ctx (~8 MB/sample) is
    cached on GPU only by the OWNING rank (`idx % world == rank`); other ranks fetch
    it from the owner via dist each step (transient, not cached), so the large ctx
    cache is SHARED across the GPUs instead of duplicated on both."""
    owner = idx % world if world > 1 else 0
    x0 = _GPU_CACHE_X.get(idx)
    if x0 is None:
        e = (_DS_CACHE or {}).get(idx) if _DS_CACHE is not None else None
        if e is not None and e.get("latents") is not None:
            lat = e["latents"]                                                  # (CH,LF,LH,LW)
            x0 = lat.permute(1, 2, 3, 0).reshape(1, seq, CH).to(device_, dtype=dtype)
        else:
            base = os.path.join(DATASET_ROOT, "latents", "scenes", f"img_{idx:03d}.pt")
            ld = torch.load(base, map_location="cpu", weights_only=True)
            lat = ld["latents"] if isinstance(ld, dict) and "latents" in ld else ld
            x0 = lat.permute(1, 2, 3, 0).reshape(1, seq, CH).to(device_, dtype=dtype)
        _GPU_CACHE_X[idx] = x0

    if rank == owner:
        ctx = _GPU_CACHE_CTX.get(idx)
        if ctx is None:
            e = (_DS_CACHE or {}).get(idx) if _DS_CACHE is not None else None
            cctx = e.get("ctx") if e is not None else None
            if cctx is None:
                cbase = os.path.join(DATASET_ROOT, "conditions", "scenes", f"img_{idx:03d}.pt")
                if os.path.exists(cbase):
                    cd = torch.load(cbase, map_location="cpu", weights_only=True)
                    cctx = cd.get("video_prompt_embeds") if isinstance(cd, dict) else None
            ctx = (cctx.unsqueeze(0).to(device_, dtype=dtype)
                   if cctx is not None else torch.zeros(1, N_CTX, DIM, device=device_, dtype=dtype))
            _GPU_CACHE_CTX[idx] = ctx
        for r in range(world):
            if r != rank:
                dist.send(_comm(ctx).contiguous(), r)
        return x0, ctx
    else:
        e = (_DS_CACHE or {}).get(idx) if _DS_CACHE is not None else None
        cctx = e.get("ctx") if e is not None else None
        ctx_shape = (1,) + tuple(cctx.shape) if cctx is not None else (1, N_CTX, DIM)
        proto = torch.zeros(ctx_shape, device=device_, dtype=dtype)
        buf = _comm_empty(proto)
        dist.recv(buf, src=owner)
        return x0, buf.to(device_).to(dtype)


def make_av_modality(device_, dtype, seed, lf=LF, lh=LH, lw=LW, seq=SEQ, fps=FPS):
    """Build video + audio Modalities (voice+face) sharing one noise level `t`.

    Returns (mod_v, target_v, mod_a, target_a). mod_a is None if no audio latent /
    audio embeds exist for this sample (audio branch skipped that step)."""
    idx = (seed % NUM_SAMPLES) if NUM_SAMPLES > 0 else seed
    x0, ctx = _load_real_sample(idx, device_, dtype, seq)
    audio_lat = _load_audio_latent(idx, device_, dtype)
    _, audio_emb = _apply_embeddings(idx, device_, dtype)

    g = torch.Generator(device=device_).manual_seed(seed)
    t = torch.rand(1, device=device_, dtype=dtype, generator=g)
    noise_v = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=g)
    x_t = (1.0 - t) * x0 + t * noise_v
    velocity_v = noise_v - x0
    timesteps_v = torch.full((1, seq), t.item(), device=device_, dtype=dtype)
    coords = _video_coords(device_, dtype, lf, lh, lw, fps)
    mod_v = Modality(enabled=True, latent=x_t, sigma=t, timesteps=timesteps_v,
                     positions=coords.contiguous(), context=ctx, context_mask=None)

    mod_a, target_a = None, None
    if audio_lat is not None and audio_emb is not None:
        ap = AudioPatchifier(1)
        T = audio_lat.shape[1]
        patched = ap.patchify(audio_lat.unsqueeze(0))          # (1,T,128)
        noise_a = torch.randn(patched.shape, device=device_, dtype=dtype, generator=g)
        x_t_a = (1.0 - t) * patched + t * noise_a
        velocity_a = noise_a - patched
        timesteps_a = torch.full((1, T), t.item(), device=device_, dtype=dtype)
        pos_a = _audio_positions(T, device_, dtype, fps)
        mod_a = Modality(enabled=True, latent=x_t_a, sigma=t, timesteps=timesteps_a,
                         positions=pos_a, context=audio_emb, context_mask=None)
        target_a = velocity_a
    return mod_v, velocity_v, mod_a, target_a


def run_shard(model, base, x, perturbations, start, end, ckpt=True, a=None):
    v = replace(base, x=x)
    for bi in range(start, end):
        v = model.block_input_processor(
            v, perturbations, bi,
            self_attn_type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
            cross_attn_type=PerturbationType.SKIP_A2V_CROSS_ATTN,
        )
        if a is not None:
            # Mirror the official forward: apply the perturbation processor to audio
            # too, so its self/cross-attn perturbation masks are set (not None).
            a = model.block_input_processor(
                a, perturbations, bi,
                self_attn_type=PerturbationType.SKIP_AUDIO_SELF_ATTN,
                cross_attn_type=PerturbationType.SKIP_V2A_CROSS_ATTN,
            )
        if a is None:
            if ckpt:
                v, _ = torch.utils.checkpoint.checkpoint(model.transformer_blocks[bi], v, None, use_reentrant=False)
            else:
                v, _ = model.transformer_blocks[bi](v, None)
        else:
            if ckpt:
                v, a = torch.utils.checkpoint.checkpoint(model.transformer_blocks[bi], v, a, use_reentrant=False)
            else:
                v, a = model.transformer_blocks[bi](v, a)
    return v.x, (a.x if a is not None else None)


class Telemetry:
    """JSON-lines telemetry writer (only last rank writes; others no-op)."""

    def __init__(self, path, enabled):
        self.path = path
        self.enabled = bool(path) and enabled
        self._fh = None
        if self.enabled and rank == world - 1:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._fh = open(path, "w", encoding="utf-8")  # truncate at run start -> no stale cross-run steps

    def emit(self, **kw):
        if self._fh is not None:
            self._fh.write(json.dumps(kw) + "\n")
            self._fh.flush()

    def close(self):
        if self._fh is not None:
            self._fh.close()


def _ffn_chunked_forward(self, x):
    """Backward-safe chunked feed-forward: split the token/sequence dim into
    chunks and run self.net per chunk, concatenating the results (no in-place
    write, so autograd works during training). Falls back to a full pass when
    the sequence is short (<= dim_threshold)."""
    if x.shape[1] > self.dim_threshold and getattr(self, "num_chunks", 1) > 1:
        n = self.num_chunks
        chunk_size = x.shape[1] // n
        outs = []
        for i in range(n):
            s = i * chunk_size
            e = (i + 1) * chunk_size if i < n - 1 else x.shape[1]
            outs.append(self.net(x[:, s:e]))
        return torch.cat(outs, dim=1)
    return self.net(x)


def _patch_ffn_chunking(model, chunks, dim_threshold, start, end):
    """Bind the backward-safe chunked forward to each owned transformer block's ff."""
    if chunks <= 1:
        return
    for bi in range(start, end):
        try:
            ff = model.transformer_blocks[bi].ff
        except Exception:
            continue
        ff.num_chunks = max(1, int(chunks))
        ff.dim_threshold = max(0, int(dim_threshold))
        ff.forward = types.MethodType(_ffn_chunked_forward, ff)
    print(f"[TR] ffn chunking chunks={chunks} threshold={dim_threshold} blocks[{start}:{end}]", flush=True)


def main():
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--config", default=None, help="path to run-config JSON")
    _a = _ap.parse_args()

    cfg = load_run_config(_a.config) if _a.config else {}

    N_STEPS = int(cfg.get("steps", 5))
    LR = float(cfg.get("lr", DEFAULT_LR))
    RANK_R = int(cfg.get("rank", 16))
    ALPHA = float(cfg.get("alpha", RANK_R))
    BUCKET = cfg.get("bucket", [W, H, F])
    tgrid = cfg.get("tile", "default")
    toverlap = int(cfg.get("overlap", 0))
    counts = cfg.get("block_counts", None)          # explicit per-GPU block counts
    ckpt_interval = int(cfg.get("checkpoint_interval", 0))
    ckpt_dir = cfg.get("checkpoint_dir", None)
    use_ckpt = bool(cfg.get("gradient_checkpointing", True))
    run_id = cfg.get("run_id", "run")
    telemetry_path = cfg.get("telemetry_path", None)
    backend = cfg.get("backend", "quanto")  # "quanto" | "bnb_nf4"
    fixed_data = bool(cfg.get("fixed_data", False))  # same latent/target every step

    global DATASET_ROOT, NUM_SAMPLES, TRAIN_AUDIO
    TRAIN_AUDIO = bool(cfg.get("train_audio", False))

    DATASET_ROOT = (cfg.get("dataset_root") or os.environ.get("LTX_DATASET_ROOT")
                    or os.path.join(os.path.dirname(_ENGINE_DIR), "dataset"))
    # Use the dataset.json sample count (the true dataset), not a count of .pt files,
    # so stale latents left over from earlier runs are never sampled.
    NUM_SAMPLES = 0
    djson = os.path.join(DATASET_ROOT, "dataset.json")
    if os.path.exists(djson):
        try:
            with open(djson, encoding="utf-8") as _f:
                NUM_SAMPLES = len(json.load(_f))
        except Exception:
            NUM_SAMPLES = 0
    if NUM_SAMPLES <= 0:
        lat_dir = os.path.join(DATASET_ROOT, "latents", "scenes")
        if os.path.isdir(lat_dir):
            NUM_SAMPLES = len([f for f in os.listdir(lat_dir) if f.endswith(".pt")])
    print(f"[TR] real-data dataset_root={DATASET_ROOT} samples={NUM_SAMPLES}", flush=True)
    t_init = time.time()
    _preload_dataset()
    dataset_preload_s = time.time() - t_init

    BW, BH, BF = int(BUCKET[0]), int(BUCKET[1]), int(BUCKET[2])
    LW = BW // 32
    LH = BH // 32
    LF = (BF - 1) // 8 + 1
    SEQ = LF * LH * LW
    tcfg = resolve_tiling(tgrid, toverlap, LH, LW)

    n_blocks = 48
    if counts:
        ranges = build_allocation(n_blocks, counts)
    else:
        ranges = shard_ranges(n_blocks, world)
    START, END = ranges[rank]

    print(f"[TR] rank{rank} {torch.cuda.get_device_name(device)} blocks[{START}:{END}] tile={tcfg.grid} ov={tcfg.overlap} backend={backend}", flush=True)
    t0 = time.time()
    if backend == "bnb_nf4":
        model = load_bnb_shard(device, rank, world, counts=counts,
                               bnb_pt=cfg.get("bnb_pt") or None)
    else:
        model = load_int4_shard(device, rank, world, counts=counts, pt_path=cfg.get("int4_pt") or None)
    model_load_s = time.time() - t0
    model.requires_grad_(False)
    t0 = time.time()
    add_lora_to_block(model, r=RANK_R, alpha=ALPHA, start=START, end=END)
    _patch_ffn_chunking(model, int(cfg.get("ffn_chunks", 1) or 1),
                        int(cfg.get("ffn_dim_threshold", 4096) or 4096), START, END)
    lora_setup_s = time.time() - t0
    trainable_this = [p for p in model.parameters() if p.requires_grad]
    t0 = time.time()
    try:
        import bitsandbytes as _bnb
        opt = _bnb.optim.AdamW8bit(trainable_this, lr=LR)
    except Exception:
        opt = torch.optim.AdamW(trainable_this, lr=LR)
    optimizer_s = time.time() - t0
    total_init_s = time.time() - t_init
    print(f"[TR] rank{rank} LoRA {sum(p.numel() for p in trainable_this)/1e6:.1f}M "
          f"(model {model_load_s:.0f}s, lora {lora_setup_s:.0f}s, opt {optimizer_s:.0f}s, data {dataset_preload_s:.0f}s)", flush=True)

    def save_lora(step):
        """Save this rank's owned-block LoRA weights as a ComfyUI-loadable .safetensors."""
        if not ckpt_dir:
            return
        os.makedirs(ckpt_dir, exist_ok=True)
        if world > 1:
            dist.barrier()
        local_sd = collect_lora_state_dict(model, start=START, end=END)
        if world > 1:
            gathered = [None] * world
            dist.all_gather_object(gathered, local_sd)
        else:
            gathered = [local_sd]
        if rank == 0:
            from safetensors.torch import save_file
            full = {}
            for g in gathered:
                full.update(g or {})
            p = os.path.join(ckpt_dir, f"lora_weights_step_{step:05d}.safetensors")
            save_file({k: v.contiguous() for k, v in full.items()}, p)
            print(f"[TR] checkpoint -> {p} ({len(full)} tensors)", flush=True)
        # Telemetry file is owned by the last rank.
        if rank == world - 1:
            total = sum(len(g or {}) for g in gathered)
            p = os.path.join(ckpt_dir, f"lora_weights_step_{step:05d}.safetensors")
            tel.emit(event="checkpoint", run_id=run_id, step=step, path=p, tensors=total)
        dist.barrier()

    from ltx_core.tools import VideoLatentTools
    video_tools = VideoLatentTools(VideoLatentPatchifier(1),
                                   VideoLatentShape(1, CH, LF, LH, LW), fps=FPS, scale_factors=SF)

    tel = Telemetry(telemetry_path, enabled=True)
    tel.emit(event="start", run_id=run_id, world=world, rank_blocks=counts,
             tile=tcfg.grid, overlap=tcfg.overlap, steps=N_STEPS,
             model_load_s=round(model_load_s, 1), lora_setup_s=round(lora_setup_s, 1),
             optimizer_s=round(optimizer_s, 1), dataset_preload_s=round(dataset_preload_s, 1),
             total_init_s=round(total_init_s, 1))

    last_loss = float("nan")
    t_last = time.time()
    # Logging cadence: the finite-check and peak-VRAM gather are pure telemetry, so
    # only run them every LOG_EVERY steps and cache the last value (removes per-step
    # GPU->CPU syncs / collectives that stall the pipeline).
    LOG_EVERY = 10
    finite = True
    peak_vram = {}
    peak_total = 0.0
    ema_dt = None
    for step in range(N_STEPS):
        t1 = time.time()
        if TRAIN_AUDIO:
            mod_v, target_v, mod_a, target_a = make_av_modality(
                device, torch.bfloat16, seed=(0 if fixed_data else step),
                lf=LF, lh=LH, lw=LW, seq=SEQ)
            mod, target = mod_v, target_v
            audio_base = model.audio_args_preprocessor.prepare(mod_a, mod_v) if mod_a is not None else None
            audio_ctx_arg = mod_a if mod_a is not None else None
        else:
            mod, target = make_modality(device, torch.bfloat16, seed=(0 if fixed_data else step),
                                        lf=LF, lh=LH, lw=LW, seq=SEQ)
            audio_base = None
            audio_ctx_arg = None
        tiles, ctxs, helper = build_tiles(mod, tcfg, video_tools)

        # ---- per-tile forward + backward, freeing each tile (peak = ONE tile) ----
        # Per-tile loss over the tile's spatial region lets us backprop and FREE each
        # tile's activations before the next, so tiling actually scales VRAM down.
        loss_v_acc = torch.zeros((), device=device, dtype=torch.float32)
        loss_a_acc = torch.zeros((), device=device, dtype=torch.float32)
        for ti, tmod in enumerate(tiles):
            base = model.video_args_preprocessor.prepare(tmod, audio_ctx_arg)
            perturbations = BatchedPerturbationConfig.empty(1, model.num_blocks, device, torch.bfloat16)
            if rank == 0:
                x_in = base.x.contiguous().clone().requires_grad_(True)
            else:
                x_in = _comm_empty(base.x)
                dist.recv(x_in, src=rank - 1)
                x_in = x_in.to(device).requires_grad_(True)
            x_in.retain_grad()
            # Audio args: built on each rank (small), only x is threaded between ranks.
            a_args = None
            a_in = None
            if audio_base is not None:
                if rank == 0:
                    a_in = audio_base.x.contiguous().clone().requires_grad_(True)
                else:
                    a_in = _comm_empty(audio_base.x)
                    dist.recv(a_in, src=rank - 1)
                    a_in = a_in.to(device).requires_grad_(True)
                a_in.retain_grad()
                a_args = replace(audio_base, x=a_in)
            # Gradient checkpointing ALWAYS follows the Train node's gradient_checkpointing
            # setting (never auto-disabled by tiling) so VRAM stays bounded on tiled runs.
            xh, ah = run_shard(model, base, x_in, perturbations, START, END, ckpt=use_ckpt, a=a_args)
            if rank < world - 1:
                dist.send(_comm(xh).contiguous(), rank + 1)
                if audio_base is not None and ah is not None:
                    dist.send(_comm(ah).contiguous(), rank + 1)

            # Per-tile loss (last rank). Audio loss added once (audio isn't tiled).
            tile_loss = None
            if rank == world - 1:
                out = model._process_output(model.scale_shift_table, model.norm_out, model.proj_out,
                                            xh, base.embedded_timestep)
                tv = target[:, ctxs[ti].keep_indices, :]
                loss_i = torch.nn.functional.mse_loss(out, tv)
                # Global-mean objective: weight each per-tile mean by its token share so
                # tiled and untiled runs report a comparable loss and unequal edge tiles
                # are not over-weighted. Sum of weights == 1 for non-overlapping tiles.
                n_tile = out.shape[1]
                tile_loss = loss_i * (n_tile / SEQ)
                loss_v_acc = loss_v_acc + loss_i.detach() * (n_tile / SEQ)
                if audio_base is not None and ti == 0 and target_a is not None:
                    aout = model._process_output(model.audio_scale_shift_table, model.audio_norm_out,
                                                 model.audio_proj_out, ah, audio_base.embedded_timestep)
                    loss_a_i = torch.nn.functional.mse_loss(aout, target_a)
                    tile_loss = tile_loss + loss_a_i
                    loss_a_acc = loss_a_acc + loss_a_i.detach()

            # ---- backward this tile, then free its activations ----
            if rank == world - 1:
                tile_loss.backward()
                g_in = x_in.grad if x_in.grad is not None else torch.zeros_like(x_in)
                if rank > 0:
                    dist.send(_comm(g_in, torch.float32).contiguous(), rank - 1)
                if audio_base is not None:
                    g_a = a_in.grad if a_in.grad is not None else torch.zeros_like(a_in)
                    dist.send(_comm(g_a, torch.float32).contiguous(), rank - 1)
                # NB: no per-tile barrier — the blocking send/recv already orders the ranks.
            else:
                g = _comm_zeros(xh, torch.float32)
                dist.recv(g, src=rank + 1)
                g = g.to(device)
                grads = torch.autograd.grad(xh, [x_in] + trainable_this, grad_outputs=g.to(xh.dtype),
                                            allow_unused=True, retain_graph=True)
                for p, gr in zip(trainable_this, grads[1:]):
                    if gr is not None:
                        p.grad = gr.detach().to(p.device) if p.grad is None else p.grad.add_(gr.detach().to(p.device))
                if rank > 0:
                    d_in = grads[0] if grads[0] is not None else torch.zeros_like(x_in)
                    dist.send(_comm(d_in, torch.float32).contiguous(), rank - 1)
                # Audio gradient channel (voice+face).
                if audio_base is not None:
                    ga = _comm_zeros(ah, torch.float32)
                    dist.recv(ga, src=rank + 1)
                    ga = ga.to(device)
                    grads_a = torch.autograd.grad(ah, [a_in] + trainable_this, grad_outputs=ga.to(ah.dtype),
                                                  allow_unused=True, retain_graph=True)
                    for p, gr in zip(trainable_this, grads_a[1:]):
                        if gr is not None:
                            p.grad = gr.detach().to(p.device) if p.grad is None else p.grad.add_(gr.detach().to(p.device))
                    if rank > 0:
                        d_a = grads_a[0] if grads_a[0] is not None else torch.zeros_like(a_in)
                        dist.send(_comm(d_a, torch.float32).contiguous(), rank - 1)
                # NB: no per-tile barrier — the blocking send/recv already orders the ranks.

            # free this tile's retained activations before the next tile (memory scales down)
            del x_in, xh
            if a_in is not None:
                del a_in, ah

        # Single host sync for loss reporting (accumulated on device across tiles).
        loss_v_sum = loss_v_acc.item()
        loss_a_sum = loss_a_acc.item()

        # Assemble losses for reporting/telemetry.
        loss_v = None
        loss_a = None
        if rank == world - 1:
            loss_v = torch.tensor(loss_v_sum, device=device, dtype=torch.float32)
            if audio_base is not None and target_a is not None:
                loss_a = torch.tensor(loss_a_sum, device=device, dtype=torch.float32)
            loss = loss_v if loss_a is None else loss_v + loss_a
        else:
            loss = torch.tensor(float("nan"), device=device, dtype=torch.float32)
        if step % LOG_EVERY == 0:
            grads_ = [p.grad for p in trainable_this if p.grad is not None]
            finite = bool(torch.all(torch.cat([torch.isfinite(g).reshape(-1) for g in grads_])).item()) if grads_ else True
        opt.step()
        opt.zero_grad()
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        dt = time.time() - t1
        now = time.time()
        sps = 1.0 / max(dt, 1e-6)
        # ETA from an EMA of true step time (not the noisy last step, not init-inflated).
        ema_dt = dt if ema_dt is None else 0.9 * ema_dt + 0.1 * dt
        eta = (N_STEPS - step - 1) * ema_dt
        # The loss is only computed on the last rank (blocks that produce the output).
        # Broadcast it so every rank reports the REAL loss (not a misleading "nan").
        if rank == world - 1:
            last_loss = loss.item()
        if world > 1:
            loss_t = torch.tensor([float(last_loss)], device=device)
            dist.broadcast(loss_t, src=world - 1)
            last_loss = loss_t.item()
        # Per-rank peak VRAM -> per-GPU; only every LOG_EVERY steps (cached otherwise).
        if step % LOG_EVERY == 0:
            peak_local = torch.tensor([torch.cuda.max_memory_allocated(device) / 1e9], device=device if USE_NCCL else torch.device("cpu"))
            peak_all = [torch.zeros(1, device=device if USE_NCCL else torch.device("cpu")) for _ in range(world)]
            dist.all_gather(peak_all, peak_local)
            peak_vram = {f"gpu{i}": round(p.item(), 2) for i, p in enumerate(peak_all)}
            peak_total = round(sum(p.item() for p in peak_all), 2)
        if rank == 0:
            print(f"[TR] step {step} tile={tcfg.grid} loss={last_loss:.6f} vram={peak_vram} total={peak_total}GB grads={finite} ({dt:.1f}s)", flush=True)
        # Telemetry file is owned by the last rank.
        if rank == world - 1:
            tel.emit(event="step", run_id=run_id, step=step, total_steps=N_STEPS, loss=last_loss,
                     loss_video=(loss_v.item() if loss_v is not None else None),
                     loss_audio=(loss_a.item() if loss_a is not None else None),
                     step_time=dt, steps_per_sec=round(sps, 3), eta_s=round(eta, 1),
                     peak_vram=peak_vram, peak_vram_gb=peak_total, grads_finite=finite, tiles=len(tiles))
        dist.barrier()

        # checkpoint (LoRA) every interval — save as ComfyUI-loadable .safetensors
        if ckpt_interval and ckpt_dir and (step + 1) % ckpt_interval == 0:
            save_lora(step + 1)

    # Always save a final checkpoint so a run shorter than the interval still
    # produces a LoRA .safetensors.
    if ckpt_dir:
        save_lora(N_STEPS)

    dist.barrier()
    if rank == world - 1:
        tel.emit(event="done", run_id=run_id, total_steps=N_STEPS, elapsed_s=round(time.time() - t0, 1))
        tel.close()
    if rank == 0:
        print(f"[TR] DONE {N_STEPS} steps in {time.time()-t0:.0f}s", flush=True)
        print("[TR] PASS" if finite else "[TR] FAIL", flush=True)


if __name__ == "__main__":
    main()
