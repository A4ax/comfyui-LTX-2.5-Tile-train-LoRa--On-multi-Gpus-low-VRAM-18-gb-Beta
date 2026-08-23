"""One-command setup for the O2noor LTX-2.5 Int4 Tile-Train ComfyUI pack.

The pack is self-contained: engine/, packages/ (vendored ltx-core / ltx-trainer),
nodes/, web/, install.py and requirements.txt all live in this folder. Install
this folder into ComfyUI's custom_nodes/, then run install.py from inside it.

It auto-detects the engine python (or makes a local venv), installs the engine
dependencies, resolves the model folder, and writes config.json next to this
file. Nothing is hardcoded to any one machine.

Usage:
  python install.py                          # create venv + install deps + write config
  python install.py --skip-install           # just resolve paths + write config
  python install.py --python PATH            # use an existing python instead of a new venv
  python install.py --install-dir <custom_nodes>  # also copy this pack into a ComfyUI custom_nodes folder
  python install.py --model-root D:\\models --no-ask
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

PACK_DIR = os.path.dirname(os.path.abspath(__file__))
REQS = os.path.join(PACK_DIR, "requirements.txt")
CONFIG_PATH = os.path.join(PACK_DIR, "config.json")
ENGINE_DIR = os.path.join(PACK_DIR, "engine")
PACKAGES_DIR = os.path.join(PACK_DIR, "packages")
VENV_PY = os.path.join(PACK_DIR, ".venv", "Scripts", "python.exe")


def detect_gpus():
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _pip(py, args):
    print("  pip " + " ".join(args), flush=True)
    r = subprocess.run([py, "-m", "pip", *args], capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        print((r.stdout or "")[-1500:])
        print((r.stderr or "")[-1500:])
        return False
    return True


def _has_nvidia():
    """True if an NVIDIA GPU is present (nvidia-smi runs OK)."""
    try:
        import subprocess as _sp
        r = _sp.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


def _install_torch(py):
    """Install torch + torchaudio with the CORRECT build.
    NVIDIA GPU present -> CUDA-enabled torch from the PyTorch CUDA index (so a fresh
    machine never ends up with CPU-only torch). Otherwise -> default (CPU) torch."""
    if not _has_nvidia():
        print("No NVIDIA GPU detected - installing CPU-only PyTorch (GPU training unavailable).", flush=True)
        ok = _pip(py, ["install", "torch", "torchaudio"])
        return ok
    print("NVIDIA GPU detected - installing CUDA-enabled PyTorch (cu124)...", flush=True)
    ok = _pip(py, ["install", "torch", "torchaudio",
                   "--index-url", "https://download.pytorch.org/whl/cu124"])
    if not ok:
        print("CUDA torch install failed; falling back to default (CPU) torch.", flush=True)
        ok = _pip(py, ["install", "torch", "torchaudio"])
    return ok


def main():
    ap = argparse.ArgumentParser(description="O2noor LTX-2.5 Int4 Tile-Train installer")
    ap.add_argument("--python", default=None, help="use an existing python instead of creating a venv")
    ap.add_argument("--skip-install", action="store_true", help="just resolve paths + write config")
    ap.add_argument("--install-dir", default=None,
                    help="copy this pack into a ComfyUI custom_nodes folder before setting up")
    ap.add_argument("--model-root", default=None, help="models root (folder containing diffusion_models/ etc.)")
    ap.add_argument("--no-ask", action="store_true", help="don't prompt; use defaults for anything unset")
    a = ap.parse_args()

    # Optional: copy this pack (and only this pack) into the target custom_nodes folder.
    work_dir = PACK_DIR
    if a.install_dir:
        dest = os.path.join(os.path.abspath(a.install_dir), os.path.basename(PACK_DIR))
        if os.path.abspath(dest) != os.path.abspath(PACK_DIR):
            os.makedirs(dest, exist_ok=True)
            for item in os.listdir(PACK_DIR):
                if item == ".venv" or item == "__pycache__":
                    continue
                s = os.path.join(PACK_DIR, item)
                if os.path.isdir(s):
                    shutil.copytree(s, os.path.join(dest, item), dirs_exist_ok=True)
                else:
                    shutil.copy2(s, os.path.join(dest, item))
            print(f"Copied pack -> {dest}")
            work_dir = dest

    sys.path.insert(0, work_dir)
    import pack_config  # resolves paths relative to the (installed) pack folder

    py = a.python if (a.python and os.path.exists(a.python)) else None
    if py is None and os.path.exists(os.path.join(work_dir, ".venv", "Scripts", "python.exe")):
        py = os.path.join(work_dir, ".venv", "Scripts", "python.exe")
    if py is None and not a.skip_install:
        print("Creating engine venv...")
        venv = os.path.join(work_dir, ".venv")
        os.makedirs(venv, exist_ok=True)
        r = subprocess.run([sys.executable, "-m", "venv", venv], capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            print((r.stderr or "")[-1500:])
            sys.exit(1)
        py = os.path.join(venv, "Scripts", "python.exe")
    if py is None or not os.path.exists(py):
        print("Could not locate/create an engine python. Pass one with --python PATH")
        sys.exit(1)
    print(f"engine python: {py}")

    if not a.skip_install:
        # Install torch/torchaudio FIRST (with the correct CUDA vs CPU build) so the
        # rest of the requirements (bitsandbytes etc.) resolve against the right torch.
        _install_torch(py)
        if os.path.exists(os.path.join(work_dir, "requirements.txt")):
            print("Installing engine requirements (this may take a while)...")
            _pip(py, ["install", "-r", os.path.join(work_dir, "requirements.txt")])

    # Write config.json (paths resolve relative to the installed pack folder).
    cfg = pack_config.load_config()
    cfg["engine_python"] = py
    cfg["engine_workdir"] = os.path.join(work_dir, "engine")
    cfg["packages_dir"] = os.path.join(work_dir, "packages")
    if a.model_root:
        cfg["models_root"] = os.path.abspath(a.model_root)
    cfg["gpus"] = detect_gpus()
    path = pack_config.save_config(cfg)
    print(f"wrote config -> {path}")
    print(f"detected GPUs: {cfg['gpus']}")
    print("""
Setup complete. Next steps:
  1. Put the models in your ComfyUI model folders:
       models/diffusion_models/ltx-2.5-22b-distilled-bnb-nf4.safetensors
       models/diffusion_models/embeddings_processor_bf16.safetensors
       models/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
       models/vae/ltx-2.5-video-vae-bf16.safetensors
       models/vae/ltx-2.5-audio-vae-bf16.safetensors
  2. Restart ComfyUI and load 'ltx25_int4_train_workflow' from the Workflow menu.
  3. GPUs are auto-detected — no GPU config needed. Use the Tile Config node only
     if you want a custom split.
""")


if __name__ == "__main__":
    main()
