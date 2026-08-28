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
    """Physical CUDA GPU count via nvidia-smi (independent of CUDA_VISIBLE_DEVICES),
    with a torch fallback."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        n = len([ln for ln in r.stdout.splitlines() if ln.strip()])
        if n > 0:
            return n
    except Exception:
        pass
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
    """Install torch + torchvision + torchaudio + torchcodec with the CORRECT build.
    NVIDIA GPU present -> CUDA-enabled torch from the PyTorch cu130 index (so a fresh
    machine never ends up with CPU-only torch, and RTX 50-series / Blackwell cards are
    supported). Otherwise -> default (CPU) torch. torchcodec is installed from the SAME
    index so its C++ ABI always matches the torch build (a mismatched wheel breaks
    torchaudio audio decode)."""
    if not _has_nvidia():
        print("No NVIDIA GPU detected - installing CPU-only PyTorch (GPU training unavailable).", flush=True)
        ok = _pip(py, ["install", "torch", "torchvision", "torchaudio", "torchcodec"])
        return ok
    print("NVIDIA GPU detected - installing CUDA-enabled PyTorch (cu130)...", flush=True)
    ok = _pip(py, ["install", "torch", "torchvision", "torchaudio", "torchcodec",
                   "--index-url", "https://download.pytorch.org/whl/cu130"])
    if not ok:
        print("CUDA torch install failed; falling back to default (CPU) torch.", flush=True)
        ok = _pip(py, ["install", "torch", "torchvision", "torchaudio", "torchcodec"])
    return ok


def _detect_ffmpeg():
    """True if the ffmpeg binary is on PATH (voice-dataset video cutting + torchaudio
    audio decode on Windows need it)."""
    try:
        if shutil.which("ffmpeg"):
            return True
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=15)
        return r.returncode == 0
    except Exception:
        return False


def _install_ffmpeg_win():
    """Try to install shared FFmpeg (BtbN) via winget on Windows; returns bool."""
    try:
        r = subprocess.run(
            ["winget", "install", "--id", "BtbN.FFmpeg.GPL.Shared", "--silent",
             "--accept-package-agreements", "--accept-source-agreements",
             "--disable-interactivity"],
            capture_output=True, text=True, timeout=300)
        return r.returncode == 0
    except Exception:
        return False


def _cl_detected():
    """True if MSVC cl.exe / VS Build Tools vcvars is available."""
    if shutil.which("cl"):
        return True
    base = r"C:\Program Files (x86)\Microsoft Visual Studio\2022"
    for sub in ("BuildTools", "Community", "Professional", "Enterprise"):
        if os.path.exists(os.path.join(base, sub, "VC", "Auxiliary", "Build", "vcvarsall.bat")):
            return True
    return False


def _check_env(py, models_root):
    """Optional --check: verify the install is ready before training. Pass/fail report."""
    print("\n=== Environment check ===", flush=True)
    ok_all = True
    # 1. NVIDIA GPU
    if _has_nvidia():
        print("  [PASS] NVIDIA GPU detected", flush=True)
    else:
        print("  [FAIL] No NVIDIA GPU - GPU training is unavailable", flush=True)
        ok_all = False
    # 2. CUDA-enabled torch
    try:
        import subprocess as _sp
        out = _sp.run([py, "-c", "import torch; print(torch.__version__, torch.cuda.is_available())"],
                      capture_output=True, text=True, timeout=120).stdout.strip()
        if not out:
            print(f"  [WARN] torch not installed in the engine python ({py}) yet - "
                  "run install.py first (without --skip-install)", flush=True)
        elif "True" in out:
            print(f"  [PASS] CUDA torch: {out}", flush=True)
        else:
            print(f"  [FAIL] torch is not CUDA-enabled: {out}", flush=True)
            ok_all = False
    except Exception as e:
        print(f"  [FAIL] could not run engine python ({e})", flush=True)
        ok_all = False
    # 3. Models present
    dm = os.path.join(models_root, "diffusion_models")
    te = os.path.join(models_root, "text_encoders")
    vae = os.path.join(models_root, "vae")
    base = os.path.join(dm, "ltx-2.5-22b-distilled-bnb-nf4.safetensors")
    if not os.path.exists(base):
        base = os.path.join(dm, "ltx-2.5-22b-distilled-int4-main-v2.safetensors")
    need = {
        "base model": base,
        "text encoder": os.path.join(te, "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
        "video vae": os.path.join(vae, "ltx-2.5-video-vae-bf16.safetensors"),
        "audio vae": os.path.join(vae, "ltx-2.5-audio-vae-bf16.safetensors"),
    }
    for label, p in need.items():
        if os.path.exists(p):
            print(f"  [PASS] {label}: {os.path.basename(p)}", flush=True)
        else:
            print(f"  [WARN] {label} missing: {p}", flush=True)
    ep = os.path.join(dm, "embeddings_processor_bf16.safetensors")
    if os.path.exists(ep):
        print("  [INFO] embeddings_processor present (optional - the base model is self-contained)", flush=True)
    # 4. FFmpeg (voice-dataset video cutting + torchaudio decode on Windows)
    if _detect_ffmpeg():
        print("  [PASS] ffmpeg detected", flush=True)
    else:
        print("  [WARN] ffmpeg not found - voice-dataset/torchaudio need it. "
              "Run:  winget install BtbN.FFmpeg.GPL.Shared", flush=True)
    # 5. VS Build Tools (informational - only needed for int4/quanto backend)
    if _cl_detected():
        print("  [INFO] cl.exe / VS Build Tools detected", flush=True)
    else:
        print("  [INFO] cl.exe NOT found - fine for the default bnb-NF4 backend (prebuilt bitsandbytes). "
              "Only needed if you use the int4/quanto backend (install VS 2022 Build Tools, C++ workload, "
              "or set LTX_VCVARS).", flush=True)
    print("=== " + ("ALL CHECKS PASS - ready to train" if ok_all else "SOME CHECKS FAILED - see above") + " ===\n", flush=True)
    return ok_all


def main():
    ap = argparse.ArgumentParser(description="O2noor LTX-2.5 Int4 Tile-Train installer")
    ap.add_argument("--python", default=None, help="use an existing python instead of creating a venv")
    ap.add_argument("--skip-install", action="store_true", help="just resolve paths + write config")
    ap.add_argument("--install-dir", default=None,
                    help="copy this pack into a ComfyUI custom_nodes folder before setting up")
    ap.add_argument("--model-root", default=None, help="models root (folder containing diffusion_models/ etc.)")
    ap.add_argument("--no-ask", action="store_true", help="don't prompt; use defaults for anything unset")
    ap.add_argument("--check", action="store_true",
                    help="verify the install is ready (GPU, CUDA torch, models) without installing")
    a = ap.parse_args()

    # Optional: copy this pack (and only this pack) into the target custom_nodes folder.
    work_dir = PACK_DIR
    if a.install_dir:
        dest = os.path.join(os.path.abspath(a.install_dir), os.path.basename(PACK_DIR))
        if os.path.abspath(dest) != os.path.abspath(PACK_DIR):
            os.makedirs(dest, exist_ok=True)
            for item in os.listdir(PACK_DIR):
                if item == ".venv" or item == "__pycache__" or item == "results":
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
    if py is None and not a.skip_install and not a.check:
        print("Creating engine venv...")
        venv = os.path.join(work_dir, ".venv")
        os.makedirs(venv, exist_ok=True)
        r = subprocess.run([sys.executable, "-m", "venv", venv], capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            print((r.stderr or "")[-1500:])
            sys.exit(1)
        py = os.path.join(venv, "Scripts", "python.exe")
    if py is None or not os.path.exists(py):
        if a.skip_install or a.check:
            py = pack_config.engine_python()  # config override -> local .venv -> current python
        else:
            print("Could not locate/create an engine python. Pass one with --python PATH")
            sys.exit(1)
    print(f"engine python: {py}")

    if a.check:
        mroot = os.path.abspath(a.model_root) if a.model_root else (pack_config.load_config().get("models_root") or "")
        _check_env(py, mroot)
        return

    if not a.skip_install:
        # Install torch/torchvision/torchaudio FIRST (with the correct CUDA vs CPU
        # build) so the rest of the requirements (bitsandbytes etc.) resolve against
        # the right torch.
        _install_torch(py)
        if os.path.exists(os.path.join(work_dir, "requirements.txt")):
            print("Installing engine requirements (this may take a while)...")
            _pip(py, ["install", "-r", os.path.join(work_dir, "requirements.txt")])
        # Optional int2/quanto backend - NON-FATAL. Default bnb-NF4 backend does not
        # need it; optimum-quanto may need VS 2022 Build Tools on Windows.
        reqs_int2 = os.path.join(work_dir, "requirements-int2.txt")
        if os.path.exists(reqs_int2):
            print("Installing optional int2/quanto extras (non-fatal)...")
            try:
                if not _pip(py, ["install", "-r", reqs_int2]):
                    print("[WARN] int2/quanto extras failed to install - the int2/qint2 backend "
                          "will be unavailable, but the default bnb-NF4 backend is unaffected. "
                          "On Windows this usually needs VS 2022 Build Tools (C++ workload) or "
                          "a matching prebuilt wheel.", flush=True)
            except Exception as e:
                print(f"[WARN] int2/quanto extras skipped ({e}). Default bnb-NF4 backend unaffected.", flush=True)

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
    if not _detect_ffmpeg():
        print("\n[WARN] ffmpeg not found on PATH. The voice-dataset steps (video cutting) and")
        print("torchaudio audio decode need it on Windows. Attempting to install shared")
        print("FFmpeg via winget...")
        if _install_ffmpeg_win():
            print("  ffmpeg installed via winget - RESTART ComfyUI (and any terminals) so the new")
            print("  PATH is picked up.")
        else:
            print("  Auto-install failed. Install it manually:")
            print('    winget install BtbN.FFmpeg.GPL.Shared')
            print("  or download from https://www.gyan.dev/ffmpeg/builds/ and add the bin folder to PATH.")
    else:
        print(f"ffmpeg detected: {shutil.which('ffmpeg')}")
    print("""
Setup complete. Next steps:
  1. Put the models in your ComfyUI model folders:
       models/diffusion_models/ltx-2.5-22b-distilled-bnb-nf4.safetensors  (or int4-main-v2)
       models/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
       models/vae/ltx-2.5-video-vae-bf16.safetensors
       models/vae/ltx-2.5-audio-vae-bf16.safetensors
     (embeddings_processor_bf16.safetensors is OPTIONAL - the base model is self-contained.)
  2. Restart ComfyUI and load 'ltx25_int4_face_voice_workflow' from the Workflow menu.
  3. GPUs are auto-detected - no GPU config needed. Use the Tile Config node only
     if you want a custom split.
  4. Run 'python install.py --check' anytime to verify GPU, CUDA torch, models and ffmpeg.
""")


if __name__ == "__main__":
    main()
