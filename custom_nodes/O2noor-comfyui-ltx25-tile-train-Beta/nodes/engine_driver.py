"""Shared engine driver: device selection + subprocess command building.

Used by Dataset / Train / Validate nodes so GPU/CPU handling is consistent and
easy for other users (laptops included).
"""
import os
import subprocess
import sys

from .. import pack_config

DEVICE_CHOICES = ["auto", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "cpu"]


def pick_device(choice: str):
    """Resolve 'auto' / 'cuda:N' / 'cpu' -> concrete device string.
    auto = first available CUDA GPU, else cpu (laptop-safe)."""
    choice = (choice or "auto").strip().lower()
    if choice == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return f"cuda:{torch.cuda.current_device()}"
        except Exception:
            pass
        return "cpu"
    if choice.startswith("cuda"):
        try:
            import torch
            if torch.cuda.is_available() and choice in (f"cuda:{i}" for i in range(torch.cuda.device_count())):
                return choice
        except Exception:
            pass
        # fall through: requested cuda but unavailable -> warn + cpu
        print(f"[LTX25] device {choice!r} requested but unavailable; using cpu", flush=True)
        return "cpu"
    if choice in ("cpu", "mps", "directml"):
        return choice
    return "cpu"


def engine_python() -> str:
    return pack_config.engine_python()


def engine_workdir() -> str:
    return pack_config.engine_workdir()


def trainer_scripts_dir() -> str:
    """Directory containing the LTX trainer CLI scripts (split_scenes, process_dataset,
    caption_videos, train). Resolved from the vendored packages_dir/ltx-trainer."""
    p = os.path.join(pack_config.packages_dir(), "ltx-trainer", "scripts")
    return p if os.path.isdir(p) else p


_FFMPEG_SHARED_BIN_CACHE = None


def ffmpeg_shared_bin():
    """Directory containing FFmpeg SHARED DLLs (avcodec etc.) so torchaudio's
    libtorchcodec decoder can load them. Auto-detects the WinGet shared build,
    overridable via config.json `ffmpeg_shared_bin`. Cached."""
    global _FFMPEG_SHARED_BIN_CACHE
    if _FFMPEG_SHARED_BIN_CACHE is not None:
        return _FFMPEG_SHARED_BIN_CACHE
    cfg = pack_config.load_config()
    cand = cfg.get("ffmpeg_shared_bin") or ""
    if cand and os.path.isdir(cand):
        _FFMPEG_SHARED_BIN_CACHE = cand
        return cand
    import glob
    base = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
    for dll in glob.glob(os.path.join(base, "**", "avcodec*.dll"), recursive=True):
        _FFMPEG_SHARED_BIN_CACHE = os.path.dirname(dll)
        return _FFMPEG_SHARED_BIN_CACHE
    _FFMPEG_SHARED_BIN_CACHE = ""
    return ""


def build_engine_cmd(script, args=None, *, visible_devices=None, scripts_dir=None):
    """Build the subprocess argv: engine python -u <scripts_dir>/<script> [args].
    Default scripts_dir = trainer_scripts_dir() so CLI tools resolve; engine scripts
    (train_parallel etc.) can pass an explicit workdir-based path.
    `visible_devices`: optional list/str of GPU indices for CUDA_VISIBLE_DEVICES.
    Returns (argv, env)."""
    py = engine_python()
    sdir = scripts_dir or trainer_scripts_dir()
    path = os.path.join(sdir, script)
    if not os.path.exists(path):
        # fall back to engine workdir (custom engine scripts)
        path = os.path.join(engine_workdir(), script)
    argv = [py, "-u", path]
    if args:
        argv += list(args)
    env = dict(os.environ)
    # Pass resolved paths to engine subprocesses so their imports/config are
    # machine-independent (scripts fall back to self-relative paths if unset).
    _set_engine_env(env)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # FFmpeg shared DLLs on PATH so torchaudio/libtorchcodec can decode audio.
    shared = ffmpeg_shared_bin()
    if shared and os.path.isdir(shared):
        env["PATH"] = shared + os.pathsep + env.get("PATH", "")
    if visible_devices:
        if isinstance(visible_devices, (list, tuple)):
            visible_devices = ",".join(str(i) for i in visible_devices)
        env["CUDA_VISIBLE_DEVICES"] = str(visible_devices)
    else:
        # keep all GPUs visible by default so cuda:N in scripts maps correctly
        env.pop("CUDA_VISIBLE_DEVICES", None)
    return argv, env


def _set_engine_env(env):
    """Inject machine-independent engine paths into a subprocess env."""
    import pack_config as _pc
    cfg = _pc.load_config()
    env["LTX_ENGINE_DIR"] = engine_workdir()
    env["LTX_ENGINE_PYTHON"] = engine_python()
    env["LTX_PACKAGES_DIR"] = _pc.packages_dir()
    env["LTX_MODELS_DIR"] = _pc.models_root()
    env["LTX_CACHE_DIR"] = cfg.get("cache_dir") or os.path.join(_pc.models_root(), "diffusion_models")
    env["LTX_DATASET_ROOT"] = cfg.get("dataset_root") or ""
    env["LTX_CONFIG"] = _pc.CONFIG_PATH


def run_engine(script, args=None, *, visible_devices=None, timeout=86400, workdir=None):
    """Run an engine script synchronously; return (returncode, stdout+stderr)."""
    argv, env = build_engine_cmd(script, args, visible_devices=visible_devices)
    wd = workdir or engine_workdir()
    print(f"[LTX25] running: {' '.join(argv)}", flush=True)
    r = subprocess.run(argv, cwd=wd, env=env, capture_output=True, text=True, timeout=timeout)
    tail = (r.stdout or "")[-3000:] + "\n" + (r.stderr or "")[-3000:]
    return r.returncode, tail
