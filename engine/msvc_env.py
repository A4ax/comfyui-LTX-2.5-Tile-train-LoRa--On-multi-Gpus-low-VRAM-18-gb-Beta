"""Load the MSVC Build Tools (vcvars64) environment into os.environ.

Reusable for all training commands so torch's C++ extension builds
(quanto tinygemm) can find cl.exe/link.exe/ninja on Windows. On non-Windows
(WSL2 / Linux) this is a no-op: the bnb/quanto JIT build uses gcc/ninja there
and needs no MSVC environment.
"""
import os
import sys


def _find_vcvars() -> str:
    # LTX_VCVARS override, then common Visual Studio Build Tools locations.
    cand = os.environ.get("LTX_VCVARS", "")
    if cand and os.path.exists(cand):
        return cand
    base = r"C:\Program Files (x86)\Microsoft Visual Studio\2022"
    for sub in ("BuildTools", "Community", "Professional", "Enterprise"):
        p = os.path.join(base, sub, "VC", "Auxiliary", "Build", "vcvarsall.bat")
        if os.path.exists(p):
            return p
    return r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"


VCVARS = _find_vcvars()


def load_msvc_env(arch: str = "x64") -> dict[str, str]:
    # Linux / WSL: no MSVC needed (gcc + ninja handle the JIT build). No-op.
    if sys.platform != "win32":
        return dict(os.environ)
    if not os.path.exists(VCVARS):
        return dict(os.environ)
    import subprocess
    import tempfile
    bat = os.path.join(tempfile.gettempdir(), f"vcvars_dump_{os.getpid()}_{arch}.bat")
    with open(bat, "w") as f:
        f.write(f'@echo off\r\ncall "{VCVARS}" {arch} >nul\r\nset\r\n')
    out = subprocess.run(["cmd", "/c", bat], capture_output=True, text=True, shell=False).stdout
    env = {}
    for line in out.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            env[k] = v
    # merge with current, but let vcvars values win for the vars it sets
    merged = dict(os.environ)
    merged.update(env)
    # ensure ninja from the engine venv is also visible
    venv_scripts = os.path.dirname(os.path.abspath(sys.executable))
    merged["PATH"] = venv_scripts + ";" + merged.get("PATH", "")
    return merged


def apply_msvc_env() -> dict[str, str]:
    env = load_msvc_env()
    os.environ.clear()
    os.environ.update(env)
    return env


if __name__ == "__main__":
    e = apply_msvc_env()
    import shutil
    print("cl:", shutil.which("cl"))
    print("ninja:", shutil.which("ninja"))
    print("link:", shutil.which("link"))
