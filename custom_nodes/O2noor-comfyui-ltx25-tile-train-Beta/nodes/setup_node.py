"""O2noorLTX25Int4Setup — resolve engine paths (easy first-time setup).

First-time users run install.py once (writes config.json); this node auto-detects
from that config and lets you override any path via widgets. Outputs the resolved
path bundle so downstream nodes (Train/Dataset/Validate) use one source of truth.
"""
import os

import folder_paths

from .. import pack_config


class O2noorLTX25Int4Setup:
    @classmethod
    def INPUT_TYPES(cls):
        cfg = pack_config.load_config()
        return {
            "required": {
                "engine_python": ("STRING", {"default": cfg.get("engine_python", "")}),
                "engine_workdir": ("STRING", {"default": cfg.get("engine_workdir", "")}),
                "int4_model_file": ("STRING", {"default": cfg.get("int4_model_file", "")}),
                "connectors_bf16": ("STRING", {"default": cfg.get("connectors_bf16", "")}),
                "text_encoder": ("STRING", {"default": cfg.get("text_encoder", "")}),
                "video_vae": ("STRING", {"default": cfg.get("video_vae", "")}),
                "save_to_config": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LTX25_SETUP",)
    RETURN_NAMES = ("setup",)
    FUNCTION = "resolve"
    CATEGORY = "ltx25-int4-train"
    TITLE = "O2noor LTX 2.5 Int4 Setup"

    def resolve(self, engine_python, engine_workdir, int4_model_file, connectors_bf16,
                text_encoder, video_vae, save_to_config=True):
        # normalize None -> ""
        vals = {k: (v or "").strip() for k, v in {
            "engine_python": engine_python, "engine_workdir": engine_workdir,
            "int4_model_file": int4_model_file, "connectors_bf16": connectors_bf16,
            "text_encoder": text_encoder, "video_vae": video_vae,
        }.items()}

        problems = []
        for k, p in vals.items():
            if p and not os.path.exists(p):
                problems.append(f"{k}: not found -> {p}")

        # write resolved paths back to config.json (optional but keeps it persistent)
        if save_to_config:
            pack_config.save_config(vals)

        ok = not problems
        print(f"[O2noorLTX25Int4Setup] engine_python={vals['engine_python']}", flush=True)
        print(f"[O2noorLTX25Int4Setup] int4_model_file={vals['int4_model_file']}", flush=True)
        print(f"[O2noorLTX25Int4Setup] connectors_bf16={vals['connectors_bf16']}", flush=True)
        if problems:
            print(f"[O2noorLTX25Int4Setup] WARN:\n  " + "\n  ".join(problems), flush=True)

        setup = {"ok": ok, "problems": problems, **vals}
        return (setup,)
