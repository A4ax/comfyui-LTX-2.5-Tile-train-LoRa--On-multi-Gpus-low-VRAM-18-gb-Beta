"""Shared pack config: auto-detect ComfyUI paths + engine setup. NEVER hardcoded.

Resolves model paths from the ComfyUI install's standard folders (diffusion_models,
text_encoders, vae) so the nodes behave like any normal ComfyUI node. All values can
be overridden by a user-editable config.json written by install.py.
"""
import json
import os
import sys

PACK_DIR = os.path.dirname(os.path.abspath(__file__))          # the node pack folder (self-contained)
CONFIG_PATH = os.path.join(PACK_DIR, "config.json")
ENGINE_DIR = os.path.join(PACK_DIR, "engine")
PACKAGES_DIR = os.path.join(PACK_DIR, "packages")


def _folder_paths():
    try:
        import folder_paths
        return folder_paths
    except Exception:
        return None


def _comfy_models_root():
    # Prefer the folder_paths the running ComfyUI actually uses (handles the
    # Desktop app's --extra-model-paths-config pointing at ComfyUI-Shared\models).
    fp = _folder_paths()
    if fp is not None:
        try:
            d = fp.get_folder_paths("diffusion_models")[0]
            if d and os.path.isdir(d):
                return os.path.dirname(d)
        except Exception:
            pass
    # fallback: sibling of the pack's ../../models (ComfyUI root/models)
    candidate = os.path.normpath(os.path.join(PACK_DIR, "..", "..", "models"))
    return candidate if os.path.isdir(candidate) else os.path.join(os.path.dirname(PACK_DIR), "models")


def _find_in(root, name):
    """Find a file by name under root (non-recursive, then one level deep)."""
    for base, _, files in os.walk(root):
        if name in files:
            return os.path.join(base, name)
        # stop after first level to keep it fast
        if base != root:
            continue
    return os.path.join(root, name)


def _venv_python(pack_dir):
    """Engine venv python for the current platform (Windows vs POSIX)."""
    if os.name == "nt":
        return os.path.join(pack_dir, ".venv", "Scripts", "python.exe")
    return os.path.join(pack_dir, ".venv", "bin", "python")


def _defaults():
    mroot = _comfy_models_root()
    dm = os.path.join(mroot, "diffusion_models")
    te = os.path.join(mroot, "text_encoders")
    vae = os.path.join(mroot, "vae")
    return {
        "engine_workdir": ENGINE_DIR,                          # self-contained engine/ inside the pack
        "engine_python": _venv_python(PACK_DIR),
        "packages_dir": PACKAGES_DIR,                          # vendored ltx-core / ltx-trainer
        "cache_dir": dm,                                       # where engine cache/derivatives land
        "int4_model_file": _find_in(dm, "ltx-2.5-22b-distilled-bnb-nf4.safetensors"),
        "connectors_bf16": _find_in(dm, "connectors_bf16.safetensors"),
        "embeddings_processor_bf16": _find_in(dm, "embeddings_processor_bf16.safetensors"),
        "text_encoder": _find_in(te, "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
                          or _find_in(te, "gemma4-12b-with-proj-ltx-2.5-Q5_K_M.gguf"),
        "video_vae": _find_in(vae, "ltx-2.5-video-vae-bf16.safetensors"),
        "audio_vae": _find_in(vae, "ltx-2.5-audio-vae-bf16.safetensors"),
        "plain_sidecar": _find_in(dm, "plain.pt"),
        # Dataset root for training data (latents/ audio_latents/ conditions/). Overridable in config.json.
        "dataset_root": os.path.join(PACK_DIR, "dataset"),
        # WSL2 / NCCL (faster multi-GPU backend). Empty = not configured yet.
        "wsl_distro": "Ubuntu",
        "wsl_python": "",          # Linux engine venv python, e.g. /home/<user>/ltx/.venv/bin/python
        "wsl_engine_dir": "",      # Linux path to the engine dir, e.g. /mnt/d/LTX-TRAINING/working
        "wsl_models_dir": "",      # Linux path to models (copy into WSL fs for speed), else /mnt/d auto-derived
    }


def load_config() -> dict:
    cfg = _defaults()
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    return cfg


def save_config(cfg: dict) -> str:
    merged = _defaults()
    merged.update(cfg)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    return CONFIG_PATH


def engine_python() -> str:
    cfg = load_config()
    if cfg.get("engine_python") and os.path.exists(cfg["engine_python"]):
        return cfg["engine_python"]
    # auto: package-local venv, else current python
    local = _venv_python(PACK_DIR)
    if os.path.exists(local):
        return local
    return sys.executable


def engine_workdir() -> str:
    cfg = load_config()
    wd = cfg.get("engine_workdir", ENGINE_DIR)
    if os.path.isdir(wd):
        return wd
    return ENGINE_DIR


def packages_dir() -> str:
    return load_config().get("packages_dir", PACKAGES_DIR)


def win_to_wsl(path):
    """Translate a Windows path (D:\\foo\\bar) to a WSL path (/mnt/d/foo/bar).
    Already-Linux paths (starting with '/') are returned unchanged. Not hardcoded
    to any specific drive — handles any single-letter drive letter."""
    if not path:
        return path
    p = str(path).replace("\\", "/")
    if p.startswith("/"):
        return p
    if len(p) >= 2 and p[1] == ":" and p[0].isalpha():
        return "/mnt/" + p[0].lower() + "/" + p[2:].lstrip("/")
    return p


def models_root() -> str:
    """Root of the ComfyUI model folders (parent of diffusion_models/ etc.).
    Honors a config.json `models_root` override; otherwise auto-detected."""
    cfg = load_config()
    if cfg.get("models_root") and os.path.isdir(cfg["models_root"]):
        return cfg["models_root"]
    return _comfy_models_root()


def media_upload_dir() -> str:
    """Where uploaded images/voices land: the SAME input dir that /view serves.

    get_input_directory() reflects the app's --input-directory (e.g. ComfyUI-Shared/input),
    which is exactly what the /view endpoint reads back. Saving uploads anywhere else makes
    thumbnails 404 (black images). Falls back to the app's first registered input folder,
    then a default input dir next to the ComfyUI root.
    """
    fp = _folder_paths()
    if fp is not None:
        try:
            d = fp.get_input_directory()
            if d:
                os.makedirs(d, exist_ok=True)
                return d
        except Exception:
            pass
        try:
            for cand in fp.get_folder_paths("input"):
                if cand and os.path.isdir(cand):
                    return cand
        except Exception:
            pass
    # fallback: default input dir next to the ComfyUI root
    d = os.path.normpath(os.path.join(PACK_DIR, "..", "..", "input"))
    os.makedirs(d, exist_ok=True)
    return d