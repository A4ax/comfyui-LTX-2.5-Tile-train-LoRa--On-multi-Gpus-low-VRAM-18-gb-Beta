"""O2noorLTX25Int4EncodeCaptions — encode captions ONCE with the 8-bit Gemma spread
across the two RTX 3060s, cache ctx embeddings, then free the GPUs.

Runs before training so the heavy Gemma never competes for VRAM. Outputs an
LTX25_CAPTIONS bundle (cache dir + index) that the Dataset node can consume
instead of re-running Gemma. The encode subprocess exits after writing, freeing
both 3060s for the training engine.
"""
import json
import os

from .. import pack_config
from . import engine_driver


class O2noorLTX25Int4EncodeCaptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LTX25_MODEL", {"tooltip": "The model from O2noorLTX25Int4LoadModel (provides text encoder + sidecar paths)."}),
                "captions": ("STRING", {
                    "multiline": True,
                    "default": "A woman with long brown hair speaks in a clear voice.",
                    "tooltip": "Captions/prompts to encode, separated by semicolons (;).",
                }),
                "gpus": ("STRING", {
                    "default": "0,1",
                    "tooltip": "GPU indices to spread the 8-bit Gemma across (default the two 3060s).",
                }),
                "output_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Where cached ctx.pt embeddings + index.json are written. Empty = the pack's captions_cache.",
                }),
                "overwrite": ("BOOLEAN", {"default": False,
                                          "tooltip": "Re-encode even if the cache index already exists."}),
            }
        }

    RETURN_TYPES = ("LTX25_CAPTIONS",)
    RETURN_NAMES = ("captions",)
    FUNCTION = "encode"
    CATEGORY = "ltx25-int4-train"
    TITLE = "O2noor LTX 2.5 Int4 Encode Captions"
    OUTPUT_NODE = False

    def encode(self, model, captions, gpus="0,1", output_dir="", overwrite=False):
        cfg = pack_config.load_config()
        if not output_dir:
            output_dir = cfg.get("captions_cache") or os.path.join(pack_config.PACK_DIR, "captions_cache")
        te = model.get("text_encoder") or cfg.get("text_encoder", "")
        sidecar = cfg.get("embeddings_processor_bf16", "") or \
            os.path.join(pack_config.models_root(), "diffusion_models", "embeddings_processor_bf16.safetensors")

        index_path = os.path.join(output_dir, "index.json")
        _dc = model.get("device_config") or {}
        if overwrite or not os.path.exists(index_path):
            rc, tail = engine_driver.run_engine("encode_captions.py", [
                "--text-encoder", te, "--sidecar", sidecar,
                "--captions", captions, "--out-dir", output_dir, "--gpus", gpus,
                "--connectors-device", _dc.get("connectors") or "gpu0"])
            ok = rc == 0
            print(f"[O2noorLTX25Int4EncodeCaptions] encode rc={rc}\n{tail}", flush=True)
        else:
            ok = True
            tail = "cache exists — reused (set overwrite to re-encode)"

        cap = {"output_dir": output_dir, "index_path": index_path, "ok": ok, "tail": tail, "gpus": gpus}
        print(f"[O2noorLTX25Int4EncodeCaptions] done ok={ok}", flush=True)
        return (cap,)
