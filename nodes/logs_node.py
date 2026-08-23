"""O2noorLTX25Int4LogsOutputs — telemetry dashboard: step pace, ETA, milestones, files.

Reads the run dir written by the Train node: telemetry.jsonl (loss / step time /
steps-per-sec / ETA / VRAM / milestones) and lists checkpoints + sample outputs.
Returns a text summary (ui text) so the pace and milestones are visible in ComfyUI.
"""
import json
import os


class O2noorLTX25Int4LogsOutputs:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "run": ("LTX25_RUN",),
                "lines": ("INT", {
                    "default": 100, "min": 1, "max": 100000, "step": 1,
                    "tooltip": "How many recent lines to show in the live log. Set high to show all.",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("summary", "milestones")
    FUNCTION = "read"
    CATEGORY = "ltx25-int4-train"
    TITLE = "O2noor LTX 2.5 Int4 Logs/Outputs"
    OUTPUT_NODE = True

    def read(self, run, lines=100):
        run_dir = run.get("run_dir", "")
        tel = run.get("telemetry_path", os.path.join(run_dir, "telemetry.jsonl"))
        ckpt = run.get("checkpoint_dir", os.path.join(run_dir, "checkpoints"))

        lines = []
        milestones = []
        if tel and os.path.exists(tel):
            with open(tel, encoding="utf-8") as f:
                raw = [json.loads(l) for l in f if l.strip()]
            for e in raw:
                if e.get("event") == "start":
                    lines.append(f"[start] {e.get('run_id')} world={e.get('world')} "
                                 f"tile={e.get('tile')} ov={e.get('overlap')} steps={e.get('steps')}")
                elif e.get("event") == "step":
                    lines.append(f"step {e.get('step'):>5}  loss={e.get('loss', float('nan')):.6f}  "
                                 f"{e.get('step_time')}s  {e.get('steps_per_sec')} step/s  "
                                 f"ETA {e.get('eta_s')}s  peak {e.get('peak_vram_gb')}GB")
                elif e.get("event") == "checkpoint":
                    milestones.append(f"checkpoint @ step {e.get('step')} -> {e.get('path')}")
                    lines.append(f"milestone: checkpoint @ step {e.get('step')}")
                elif e.get("event") == "done":
                    lines.append(f"[done] {e.get('total_steps')} steps in {e.get('elapsed_s')}s")
            # pace summary
            steps = [e for e in raw if e.get("event") == "step"]
            if len(steps) >= 2:
                times = [e["step_time"] for e in steps]
                avg = sum(times) / len(times)
                lines.append(f"[pace] avg step {avg:.2f}s -> {1.0/avg:.3f} step/s; "
                             f"last loss {steps[-1].get('loss')}")
        else:
            lines.append("no telemetry yet (run may still be starting)")

        ckpt_files = []
        if os.path.isdir(ckpt):
            ckpt_files = sorted(f for f in os.listdir(ckpt) if f.endswith(".safetensors"))
        if ckpt_files:
            milestones.append("checkpoints: " + ", ".join(ckpt_files))

        summary = "\n".join(lines[-25:])
        milestone_text = "\n".join(milestones)
        print(f"[O2noorLTX25Int4LogsOutputs]\n{summary}\n-- milestones --\n{milestone_text}", flush=True)
        return {"ui": {"text": [summary]}, "result": (summary, milestone_text)}
