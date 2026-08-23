"""Shared, machine-independent path setup for engine scripts.

Engine scripts live in <pack>/engine. From here we can always reach:
  - sibling engine modules (msvc_env, int4_parallel, tiling_feature, ...)
  - the vendored packages at <pack>/packages (ltx-core, ltx-trainer, ...)
  - config.json at <pack>/config.json

Everything is derived relative to this file (or overridden by LTX_* env vars set
by the ComfyUI nodes), so the same pack works on any machine / any ComfyUI path.
"""
import os
import sys

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
PACK_ROOT = os.path.dirname(ENGINE_DIR)                        # self-contained pack folder
CONFIG_PATH = os.environ.get("LTX_CONFIG") or os.path.join(PACK_ROOT, "config.json")
PACKAGES_DIR = os.environ.get("LTX_PACKAGES_DIR") or os.path.join(PACK_ROOT, "packages")
MODELS_ROOT = os.environ.get("LTX_MODELS_DIR") or os.path.normpath(os.path.join(PACK_ROOT, "..", "..", "models"))
CACHE_DIR = os.environ.get("LTX_CACHE_DIR") or os.path.join(MODELS_ROOT, "diffusion_models")
DATASET_ROOT = os.environ.get("LTX_DATASET_ROOT") or os.path.join(PACK_ROOT, "dataset")


def setup_paths():
    """Add the engine dir + vendored packages to sys.path (idempotent)."""
    if ENGINE_DIR not in sys.path:
        sys.path.insert(0, ENGINE_DIR)
    for pkg in ("ltx-core", "ltx-trainer", "ltx-pipelines", "ltx-kernels"):
        src = os.path.join(PACKAGES_DIR, pkg, "src")
        if os.path.isdir(src) and src not in sys.path:
            sys.path.insert(0, src)
    return ENGINE_DIR


def load_config():
    import json
    cfg = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            pass
    return cfg
