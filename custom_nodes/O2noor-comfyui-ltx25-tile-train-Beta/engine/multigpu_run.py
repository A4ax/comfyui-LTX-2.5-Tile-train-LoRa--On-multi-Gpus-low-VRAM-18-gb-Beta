"""Custom multi-GPU launcher for the TDP training engine.

Bypasses `accelerate launch` (torch 2.13 on Windows requests libuv for its
rendezvous store, which this build lacks). Spawns one process per GPU,
initializes torch.distributed with the classic gloo/file TCP store, then
runs the target script. Each rank gets LOCAL_RANK / RANK / WORLD_SIZE and
CUDA_VISIBLE_DEVICES pinned to its GPU.

Usage:
  python working/multigpu_run.py --world 2 --config working/ddp_2gpu.yaml target_script.py [args...]
"""
import argparse
import os
import subprocess
import sys

import yaml

VENV_PY = os.environ.get("LTX_ENGINE_PYTHON") or sys.executable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, required=True, help="number of processes/GPUs")
    ap.add_argument("--config", required=True, help="accelerate-style yaml (num_processes, mixed_precision)")
    ap.add_argument("--devices", default=None, help="comma list of GPU indices to use (default: 0..world-1)")
    ap.add_argument("--script", required=True)
    ap.add_argument("script_args", nargs="*")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mixed = cfg.get("mixed_precision", "bf16")
    os.environ["MIXED_PRECISION"] = str(mixed)
    os.environ["LTX_DISTRIBUTED"] = "1"

    devices = (args.devices or ",".join(str(i) for i in range(args.world))).split(",")
    assert len(devices) == args.world, "devices count must equal world"

    procs = []
    for rank in range(args.world):
        env = dict(os.environ)
        env["RANK"] = str(rank)
        env["LOCAL_RANK"] = str(rank)
        env["WORLD_SIZE"] = str(args.world)
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = "29500"
        # Keep ALL GPUs visible so DDP's device-index bookkeeping works; each rank
        # pins to its own device via LOCAL_RANK. (Setting CUDA_VISIBLE_DEVICES per
        # rank breaks DDP's _streams[device.index] lookup on this torch build.)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env.pop("CUDA_VISIBLE_DEVICES", None)
        # torch 2.13 on this Windows build has no libuv; force the classic TCP store.
        env["USE_LIBUV"] = "0"
        cmd = [VENV_PY, "-u", args.script, *args.script_args]
        procs.append(subprocess.Popen(cmd, env=env))

    rc = 0
    for p in procs:
        if p.wait() != 0:
            rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
