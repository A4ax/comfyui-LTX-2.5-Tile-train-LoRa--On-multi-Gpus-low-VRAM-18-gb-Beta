"""Merge the per-file bnb-NF4 cache into ONE self-contained .safetensors for hosting.

Structure (keys that load_bnb_shard reads):
  - transformer_blocks.N.<sub>            bnb-NF4 block state_dicts (flat incl. weight.* meta)
  - <aux_name>.<sub>                      aux modules, bf16
  - <bare_param>                          the 3 top-level bare params, bf16
  - metadata header "int4_config"         skeleton rebuild config (no separate config.json)

Output: ltx-2.5-22b-distilled-bnb-nf4.safetensors (~10.5GB). Loads straight to GPU via
safe_open(device=...); per-rank block slicing means only owned blocks are materialized.
"""
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import os  # noqa: E402
import torch  # noqa: E402
import engine_env  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

CACHE = os.path.join(engine_env.MODELS_ROOT, "diffusion_models", "ltx-bnb-nf4-cache")
DST = os.path.join(engine_env.MODELS_ROOT, "diffusion_models", "ltx-2.5-22b-distilled-bnb-nf4.safetensors")


def main():
    t0 = time.time()
    merged = {}
    n = 0

    def add(prefix, sd):
        nonlocal n
        for k, v in sd.items():
            merged[f"{prefix}{k}"] = v.contiguous()
        n += len(sd)

    aux = sorted(x for x in os.listdir(CACHE) if x.startswith("aux_") and x.endswith(".pt"))
    for fn in aux:
        name = fn[len("aux_"):-len(".pt")]
        add(f"{name}.", torch.load(os.path.join(CACHE, fn), map_location="cpu", weights_only=False))
        print(f"[MERGE] aux {name} (+{len([k for k in merged if k.startswith(name)])}) ({time.time()-t0:.0f}s)", flush=True)

    bp = os.path.join(CACHE, "bare.pt")
    if os.path.exists(bp):
        add("", torch.load(bp, map_location="cpu", weights_only=False))
        print(f"[MERGE] bare ({time.time()-t0:.0f}s)", flush=True)

    for bi in range(48):
        add(f"transformer_blocks.{bi}.",
            torch.load(os.path.join(CACHE, f"block_{bi:02d}.pt"), map_location="cpu", weights_only=False))
        if bi % 8 == 7:
            print(f"[MERGE] blocks ...{bi+1}/48 ({len(merged)} tensors) ({time.time()-t0:.0f}s)", flush=True)

    with open(os.path.join(CACHE, "config.json"), encoding="utf-8") as fh:
        meta = json.load(fh).get("int4_config", {})

    print(f"[MERGE] saving {len(merged)} tensors -> {DST}...", flush=True)
    save_file(merged, DST, metadata={"int4_config": json.dumps(meta, default=str)})
    print(f"[MERGE] DONE {os.path.getsize(DST)/1e9:.1f}GB in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
