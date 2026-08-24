"""Build a SELF-CONTAINED single int2 .safetensors file for hosting.

Reads a per-module int2 cache (config.json, aux_*.pt, bare.pt, block_*.pt) and
merges it into ONE self-contained .safetensors that the engine's load_int4_shard
can load directly (no cache folder needed at runtime). The int4_config is embedded
as file metadata so the loader reads it straight from the file.

Usage:
  python build_int2_selfcontained.py --cache-dir <dir> --out <file.safetensors> [--device cuda:0]

No hardcoded paths. Merge runs on GPU (--device) to avoid the old CPU-RAM path.
"""
import argparse
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

PLAIN = {"patchify_proj", "audio_patchify_proj", "proj_out", "audio_proj_out"}


def main():
    ap = argparse.ArgumentParser(description="Build a self-contained int2 .safetensors from a per-module int2 cache")
    ap.add_argument("--cache-dir", required=True, help="int2 cache dir (config.json, aux_*.pt, bare.pt, block_*.pt)")
    ap.add_argument("--out", required=True, help="output .safetensors path")
    ap.add_argument("--device", default=os.environ.get("LTX_BUILD_DEVICE", "cuda:0"),
                    help="device to load/merge on (default cuda:0)")
    a = ap.parse_args()

    cache = a.cache_dir
    device = a.device
    t0 = time.time()
    merged = {}
    n = 0

    def add(prefix, sd):
        nonlocal n
        for k, v in sd.items():
            merged[f"{prefix}{k}"] = v
        n += len(sd)

    # config -> file metadata (so load_int4_shard reads it from the file, no cache needed)
    meta = {}
    cfg_path = os.path.join(cache, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as fh:
            meta = {"int4_config": fh.read()}
    else:
        print("[V2i2] WARNING: config.json not found in cache - output will have no int4_config metadata", flush=True)

    # plain modules (bf16)
    for name in PLAIN:
        p = os.path.join(cache, f"aux_{name}.pt")
        if os.path.exists(p):
            sd = torch.load(p, map_location=device, weights_only=False)
            add(f"{name}.", sd)
            n += len(sd)
            print(f"[V2i2] plain {name} (+{len(sd)}) ({time.time()-t0:.0f}s)", flush=True)

    # aux (non-plain, quantized)
    for fn in sorted(x for x in os.listdir(cache) if x.startswith("aux_") and x.endswith(".pt")):
        name = fn[len("aux_"):-len(".pt")]
        if name in PLAIN:
            continue
        sd = torch.load(os.path.join(cache, fn), map_location=device, weights_only=False)
        add(f"{name}.", sd)
        print(f"[V2i2] aux {name} (+{len(sd)}) ({time.time()-t0:.0f}s)", flush=True)

    # bare
    bp = os.path.join(cache, "bare.pt")
    if os.path.exists(bp):
        bare = torch.load(bp, map_location=device, weights_only=False)
        add("", bare)
        print(f"[V2i2] bare (+{len(bare)}) ({time.time()-t0:.0f}s)", flush=True)

    # blocks (auto-counted, not hardcoded)
    block_files = sorted(x for x in os.listdir(cache) if x.startswith("block_") and x.endswith(".pt"))
    for fn in block_files:
        bi = int(fn[len("block_"):-len(".pt")])
        sd = torch.load(os.path.join(cache, fn), map_location=device, weights_only=False)
        add(f"transformer_blocks.{bi}.", sd)
        if (bi + 1) % 8 == 0:
            print(f"[V2i2] blocks ...{bi+1}/{len(block_files)} ({time.time()-t0:.0f}s)", flush=True)

    print(f"[V2i2] saving {len(merged)} tensors -> {a.out}...", flush=True)
    save_file({k: v.contiguous() for k, v in merged.items()}, a.out, metadata=meta)
    print(f"[V2i2] DONE {os.path.getsize(a.out)/1e9:.1f}GB in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
