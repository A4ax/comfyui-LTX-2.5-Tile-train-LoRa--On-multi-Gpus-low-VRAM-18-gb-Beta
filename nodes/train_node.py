"""O2noorLTX25Int4Train — main training node.

Consumes the LTX25_MODEL (LoadModel), LTX25_TILECONFIG (TileConfig) and
LTX25_DATASET (Dataset, which carries the resolution) and trains the int4 LoRA
on the chosen GPUs with the chosen tiling. No duplicate resolution input — it
comes from the Dataset node.
"""
import json
import os
import subprocess
import time

import folder_paths

from .. import pack_config
from . import engine_driver


class O2noorLTX25Int4Train:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LTX25_MODEL", {"tooltip": "The int4 model from O2noorLTX25Int4LoadModel."}),
                "dataset": ("LTX25_DATASET", {"tooltip": "Preprocessed dataset (provides resolution + media)."}),
                "run_name": ("STRING", {"default": "ltx25_train",
                                        "tooltip": "Base name for this run. With auto_unique on, a timestamp is "
                                                   "appended when the folder already exists so retraining never "
                                                   "overwrites a previous LoRA."}),
                "auto_unique": ("BOOLEAN", {"default": True,
                    "tooltip": "If the run_name output folder already exists, append a timestamp to make a "
                               "unique run dir (never overwrites a previous LoRA). Turn off to reuse the exact "
                               "run_name folder."}),
                "steps": ("INT", {"default": 2000, "min": 1, "max": 100000, "step": 1,
                                  "tooltip": "Number of training steps."}),
                "lr": ("FLOAT", {"default": 0.0003, "min": 1e-7, "max": 0.1, "step": 1e-6,
                                 "tooltip": "Learning rate."}),
                "rank": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1,
                                 "tooltip": "LoRA rank (adapter size)."}),
                "alpha": ("FLOAT", {"default": 16.0, "min": 0.0, "max": 512.0, "step": 0.5,
                                    "tooltip": "LoRA alpha (scaling)."}),
                "checkpoint_interval": ("INT", {"default": 250, "min": 0, "max": 100000, "step": 1,
                                                "tooltip": "Save a LoRA checkpoint every N steps (0 = off)."}),
                "blocking": ("BOOLEAN", {"default": True,
                                  "tooltip": "Run in the foreground (blocks) so ComfyUI waits for completion."}),
                "checkpoint_level": (["off", "auto", "on"], {"default": "on",
                    "tooltip": "Gradient-checkpointing behavior.\n"
                               "  on   = checkpoint every block (DEFAULT, most VRAM-safe): the smallest live "
                               "VRAM footprint and the reliable choice — keep it ON. The bnb-NF4 backward "
                               "allocates transient dequantize tensors (up to ~256 MB-1 GB); with off/auto "
                               "leaving blocks recompute-on-demand the allocator fragments and a transient "
                               "OOMs even when live VRAM is far below the card limit.\n"
                               "  auto = selective per-block (speed/VRAM balance): keeps OFF speed on cards "
                               "that fit, checkpoints only the blocks that won't fit a small GPU (e.g. the "
                               "4060). Faster than on, but still can fragment a big GPU — only if you "
                               "need the speed.\n"
                               "  off  = no checkpointing (fastest, most VRAM). Use only when all GPUs are "
                               "large enough (e.g. all 12 GB) — otherwise a small GPU may OOM."}),
                "use_wsl": ("BOOLEAN", {"default": False,
                    "tooltip": "Run the engine inside WSL2 (Linux) to use the faster NCCL multi-GPU backend.\n"
                               "  OFF = current native-Windows engine (gloo).\n"
                               "  ON  = launches via WSL2 with NCCL (faster GPU-GPU communication).\n"
                               "Requires WSL2 + a configured Linux engine venv (wsl_python / wsl_engine_dir in "
                               "config.json). See README -> WSL / NCCL."}),
            },
            "optional": {
                "tile_config": ("LTX25_TILECONFIG", {
                    "tooltip": "Optional advanced tiling/GPU-split config. Omitted = auto split across the 2x RTX 3060s.",
                }),
            }
        }

    RETURN_TYPES = ("LTX25_RUN",)
    RETURN_NAMES = ("run",)
    FUNCTION = "run"
    CATEGORY = "ltx25-int4-train"
    TITLE = "O2noor LTX 2.5 Int4 Train"

    def run(self, model, dataset, run_name, auto_unique, steps, lr, rank, alpha,
            checkpoint_interval, blocking, checkpoint_level, use_wsl=False,
            tile_config=None):
        workdir = engine_driver.engine_workdir()
        py = engine_driver.engine_python()

        tcfg = tile_config or {}
        tile = f"{tcfg.get('vertical_tiles', 1)}x{tcfg.get('horizontal_tiles', 1)}"
        overlap = tcfg.get("overlap", 0)
        block_counts = tcfg.get("block_counts", None)
        int4_model_file = tcfg.get("int4_model_file", "") or model.get("int4_model_file", "")

        # Transformer device placement (BLOCKS per GPU) from the Load Model node overrides
        # the TileConfig block split. The user picks how many of the 48 blocks go to each GPU.
        dc = (model.get("device_config") or {}).get("transformer") or {}
        counts = []
        for k in sorted(dc.keys(), key=lambda k: int(k[10:]) if k.startswith("blocks_gpu") and k[10:].isdigit() else 0):
            if k.startswith("blocks_gpu") and k[10:].isdigit():
                counts.append(int(dc.get(k) or 0))
        if counts and any(c > 0 for c in counts):
            n_blocks = 48
            total = sum(counts)
            if total < n_blocks:
                first = next((i for i, c in enumerate(counts) if c > 0), 0)
                counts[first] += n_blocks - total
            elif total > n_blocks:
                scale = n_blocks / total
                counts = [max(0, int(c * scale)) for c in counts]
                counts[0] += n_blocks - sum(counts)
            block_counts = [c for c in counts if c > 0]
            print(f"[O2noorLTX25Int4Train] device_config transformer -> blocks/GPU={block_counts}", flush=True)

        # Auto-detect the quant backend from the model filename:
        # "bnb" -> bitsandbytes NF4 (fast fused 4-bit kernels); otherwise quanto.
        base = os.path.basename(int4_model_file).lower()
        backend = "bnb_nf4" if "bnb" in base else "quanto"

        # resolution comes from the Dataset output
        bucket = [int(dataset.get("width", 512)), int(dataset.get("height", 512)),
                  int(dataset.get("frames", 17))]

        out_root = folder_paths.get_output_directory()
        run_dir = os.path.join(out_root, run_name)
        if auto_unique and os.path.exists(run_dir):
            run_dir = os.path.join(out_root, f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(run_dir, exist_ok=True)
        actual_run_id = os.path.basename(run_dir)
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        tel_path = os.path.join(run_dir, "telemetry.jsonl")
        print(f"[O2noorLTX25Int4Train] run_dir={run_dir} auto_unique={bool(auto_unique)}", flush=True)

        run_cfg = {
            "run_id": actual_run_id,
            "steps": steps,
            "lr": lr,
            "rank": rank,
            "alpha": alpha,
            "bucket": bucket,
            "tile": tile,
            "overlap": overlap,
            "world": len(block_counts) if block_counts else 2,
            "block_counts": block_counts,
            "checkpoint_interval": checkpoint_interval,
            "checkpoint_dir": ckpt_dir,
            "telemetry_path": tel_path,
            "int4_cache": os.path.dirname(int4_model_file) if int4_model_file else "",
            "int4_pt": int4_model_file or None,
            "backend": backend,
            "bnb_pt": int4_model_file if backend == "bnb_nf4" else None,
            "dataset_root": dataset.get("dataset_root", ""),
            "device_config": model.get("device_config") or {},
            "checkpoint_level": str(checkpoint_level or "on"),
            # Backward-compat placeholder (engine now reads checkpoint_level above).
            "gradient_checkpointing": bool(str(checkpoint_level or "on") == "on"),
            # Optional low-VRAM feed-forward chunking (from O2noorLTX25ChunkFeedForward,
            # stamped on the model input).
            "ffn_chunks": int(model.get("ffn_chunks", 1) or 1),
            "ffn_dim_threshold": int(model.get("ffn_dim_threshold", 4096) or 4096),
            # Voice+face (audio) training: enabled when the dataset was built in "face+voice" mode.
            "train_audio": bool(dataset.get("use_audio", False)),
            "audio_vae": dataset.get("audio_vae", "") or model.get("audio_vae", ""),
            "embeddings_processor_bf16": model.get("embeddings_processor_bf16", ""),
            "text_encoder": model.get("text_encoder", ""),
            # Dataset build info (so the Summary node can display it).
            "mode": dataset.get("mode", "face-only"),
            "use_audio": bool(dataset.get("use_audio", False)),
            "segment_duration": dataset.get("segment_duration"),
        }
        run_json_win = os.path.join(run_dir, "run.json")

        # Auto-detect the user's GPUs — never hardcode. Use all available CUDA
        # devices (capped by the block split if one was provided via Tile Config).
        import torch as _torch
        try:
            n_gpu = _torch.cuda.device_count() if _torch.cuda.is_available() else 0
        except Exception:
            n_gpu = 0
        world = min(run_cfg["world"] or n_gpu, n_gpu) if n_gpu else 1
        world = max(1, world)
        devices = ",".join(str(i) for i in range(world))

        if use_wsl:
            # ---- WSL2 / NCCL backend --------------------------------------------
            # Translate every path in run_cfg to WSL form (/mnt/d/...) so the Linux
            # engine can read/write them, and write run.json with those paths.
            wsl_cfg = pack_config.load_config()
            wsl_py = wsl_cfg.get("wsl_python") or ""
            wsl_engine = wsl_cfg.get("wsl_engine_dir") or ""
            wsl_distro = wsl_cfg.get("wsl_distro") or "Ubuntu"
            if not wsl_py or not wsl_engine:
                raise ValueError(
                    "use_wsl=ON requires wsl_python and wsl_engine_dir in config.json "
                    "(see README -> WSL / NCCL).")
            def _t(v):
                if isinstance(v, str):
                    return pack_config.win_to_wsl(v)
                if isinstance(v, list):
                    return [pack_config.win_to_wsl(x) if isinstance(x, str) else x for x in v]
                return v
            run_cfg_wsl = {k: _t(v) for k, v in run_cfg.items()}
            with open(run_json_win, "w", encoding="utf-8") as f:
                json.dump(run_cfg_wsl, f, indent=2)
            run_json = pack_config.win_to_wsl(run_json_win)

            _we = wsl_engine.rstrip("/") + "/"
            argv = ["wsl", "-d", wsl_distro, "-e", wsl_py, "-u",
                    _we + "multigpu_run.py",
                    "--world", str(world), "--devices", devices,
                    "--config", _we + "ddp_2gpu.yaml",
                    "--script", _we + "train_parallel.py",
                    "--", "--config", run_json]
            env = dict(os.environ)
            env["LTX_ENGINE_PYTHON"] = wsl_py
            env["LTX_ENGINE_DIR"] = wsl_engine
            env["LTX_PACKAGES_DIR"] = pack_config.win_to_wsl(wsl_cfg.get("packages_dir") or "")
            env["LTX_MODELS_DIR"] = wsl_cfg.get("wsl_models_dir") or pack_config.win_to_wsl(wsl_cfg.get("cache_dir") or "")
            env["LTX_CACHE_DIR"] = env["LTX_MODELS_DIR"]
            env["LTX_DATASET_ROOT"] = pack_config.win_to_wsl(run_cfg.get("dataset_root") or "")
            env["LTX_CONFIG"] = run_json
            print(f"[O2noorLTX25Int4Train] use_wsl=ON distro={wsl_distro} (NCCL backend)", flush=True)
        else:
            # ---- native Windows engine (gloo) -----------------------------------
            run_json = run_json_win
            with open(run_json_win, "w", encoding="utf-8") as f:
                json.dump(run_cfg, f, indent=2)
            argv = [py, "-u", os.path.join(workdir, "multigpu_run.py"),
                    "--world", str(world), "--devices", devices,
                    "--config", os.path.join(workdir, "ddp_2gpu.yaml"),
                    "--script", os.path.join(workdir, "train_parallel.py"),
                    "--", "--config", run_json]
            env = dict(os.environ)
            engine_driver._set_engine_env(env)
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(n_gpu))

        print(f"[O2noorLTX25Int4Train] run_dir={run_dir}", flush=True)
        print(f"[O2noorLTX25Int4Train] bucket={bucket} tile={tile} ov={overlap} blocks/GPU={block_counts}", flush=True)
        # Tell the frontend where telemetry lives so the live Logs widget can poll
        # DURING the (blocking) run instead of only after it finishes.
        try:
            import server
            server.PromptServer.instance.send_sync(
                "ltx25:telemetry",
                {"telemetry_path": tel_path, "run_dir": run_dir, "run_name": run_name})
        except Exception as e:
            print(f"[O2noorLTX25Int4Train] (telemetry notify skipped: {e})", flush=True)
        t0 = time.time()
        if blocking:
            r = subprocess.run(argv, cwd=workdir, env=env, timeout=86400)
            rc = r.returncode
            tail = ""
        else:
            subprocess.Popen(argv, cwd=workdir, env=env)
            rc = None
            tail = "launched in background"

        run = {
            "run_dir": run_dir,
            "run_json": run_json_win,
            "telemetry_path": tel_path,
            "checkpoint_dir": ckpt_dir,
            "lora_path": os.path.join(ckpt_dir, "lora_weights_step_XXXXX.safetensors"),
            "rc": rc,
            "elapsed_s": round(time.time() - t0, 1),
            "world": world,
            "block_counts": block_counts,
            "tile": tile,
            "overlap": overlap,
            "bucket": bucket,
            "tail": tail,
        }
        print(f"[O2noorLTX25Int4Train] finished rc={rc} in {time.time()-t0:.0f}s", flush=True)
        return (run,)
