"""O2noorLTX25Int4Validate — post-train sample rendering (in-dist / OOD / held-out).

Loads the trained LoRA checkpoint and renders samples via the engine's
validation helper. Currently the engine's synthetic forward is the proof path;
this node shells to a small render script when present and otherwise reports the
checkpoint + prompts for manual validation. Kept as a driver so it slots into the
same workflow.
"""
import os
import subprocess

from .. import pack_config


class O2noorLTX25Int4Validate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "run": ("LTX25_RUN",),
                "prompts": ("STRING", {
                    "multiline": True,
                    "default": "A woman with long brown hair speaks in a clear voice.",
                }),
                "inference_steps": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 32,
                                  "tooltip": "Render width (multiple of 32)."}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 32,
                                   "tooltip": "Render height (multiple of 32)."}),
                "frames": ("INT", {"default": 25, "min": 9, "max": 129, "step": 8,
                                   "tooltip": "Render frames (must be 1 mod 8: 17, 25, 33...)."}),
                "generate_audio": ("BOOLEAN", {"default": True,
                                               "tooltip": "Include the learned voice track in the rendered sample (higher VRAM/time)."}),
            },
            "optional": {
                "setup": ("LTX25_SETUP",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("validation",)
    FUNCTION = "validate"
    CATEGORY = "ltx25-int4-train"
    TITLE = "O2noor LTX 2.5 Int4 Validate"
    OUTPUT_NODE = True

    def validate(self, run, prompts, inference_steps=20, width=512, height=512, frames=25,
                 generate_audio=True, setup=None):
        cfg = pack_config.load_config()
        if setup and setup.get("engine_workdir"):
            cfg.update({k: v for k, v in setup.items() if k in ("engine_workdir",)})
        workdir = cfg.get("engine_workdir") or engine_driver.engine_workdir()
        py = cfg.get("engine_python") or engine_driver.engine_python()

        ckpt_dir = run.get("checkpoint_dir", "")
        ckpt_files = []
        if os.path.isdir(ckpt_dir):
            ckpt_files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".safetensors"))
        latest = os.path.join(ckpt_dir, ckpt_files[-1]) if ckpt_files else ""

        # If a render script exists in the engine, drive it; otherwise report.
        render_script = os.path.join(workdir, "render_lora_samples.py")
        out = []
        if latest and os.path.exists(render_script):
            out_dir = os.path.join(run.get("run_dir", ""), "samples")
            os.makedirs(out_dir, exist_ok=True)
            cmd = [py, "-u", render_script, "--checkpoint", latest,
                   "--out-dir", out_dir, "--prompts", prompts,
                   "--inference-steps", str(inference_steps),
                   "--width", str(width), "--height", str(height), "--frames", str(frames),
                   "--generate-audio", "1" if generate_audio else "0"]
            print(f"[O2noorLTX25Int4Validate] {' '.join(cmd)}", flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=86400)
            out.append(f"render rc={r.returncode}")
            out.append((r.stdout or "")[-2000:])
            out.append((r.stderr or "")[-2000:])
        elif latest:
            out.append(f"checkpoint: {latest}")
            out.append("render_lora_samples.py not present in engine workdir — no render performed.")
        else:
            out.append("no checkpoints found yet — train first.")

        text = "\n".join(out)
        print(f"[O2noorLTX25Int4Validate]\n{text}", flush=True)
        return {"ui": {"text": [text]}, "result": (text,)}
