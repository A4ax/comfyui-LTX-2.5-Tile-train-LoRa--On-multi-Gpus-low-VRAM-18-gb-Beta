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


def _detect_gpus():
    """Number of CUDA-visible GPUs (respects CUDA_VISIBLE_DEVICES)."""
    try:
        import torch
        n = torch.cuda.device_count()
        return max(1, n)
    except Exception:
        return 1


def _default_layer_split(n):
    """Default Gemma layer split per GPU: 1->[48], 2->[24,24], 3+->fills first n-1
    evenly, last = 0 (mirrors the Load Model node's transformer-block default)."""
    if n <= 1:
        return [48]
    if n == 2:
        return [24, 24]
    used = n - 1
    base = 48 // used
    rem = 48 % used
    layers = [base + (1 if i < rem else 0) for i in range(used)] + [0]
    return layers


class O2noorLTX25Int4EncodeCaptions:
    @classmethod
    def INPUT_TYPES(cls):
        n_gpu = _detect_gpus()
        defaults = _default_layer_split(n_gpu)
        required = {
            "model": ("LTX25_MODEL", {"tooltip": "The model from O2noorLTX25Int4LoadModel (provides text encoder + sidecar paths)."}),
            "captions": ("STRING", {
                "multiline": True,
                "default": "A woman with long brown hair speaks in a clear voice.",
                "tooltip": "Captions/prompts to encode, separated by semicolons (;).",
            }),
            "gpus": ("STRING", {
                "default": "0,1",
                "tooltip": "GPU indices to spread the 8-bit Gemma across, in the SAME ORDER as the "
                           "layer sliders below (e.g. '0,1,2' or '0,2').",
            }),
            "output_dir": ("STRING", {
                "default": "",
                "tooltip": "Where cached ctx.pt embeddings + index.json are written. Empty = the pack's captions_cache.",
            }),
            "overwrite": ("BOOLEAN", {"default": False,
                                      "tooltip": "Re-encode even if the cache index already exists."}),
        }
        # ---- Gemma transformer LAYERS per GPU (n_layers = 48), one slider per GPU ----
        for i in range(n_gpu):
            required[f"layers_gpu{i}"] = ("INT", {"min": 0, "max": 48, "step": 1,
                "default": defaults[i],
                "tooltip": f"How many of the 48 Gemma layers go to GPU {i} (index {i} in the `gpus` list). "
                           "Must total <= 48; the node auto-fills the remainder. 0 = GPU not used for layers."})
        return {"required": required}

    RETURN_TYPES = ("LTX25_CAPTIONS",)
    RETURN_NAMES = ("captions",)
    FUNCTION = "encode"
    CATEGORY = "ltx25-int4-train"
    TITLE = "O2noor LTX 2.5 Int4 Encode Captions"
    OUTPUT_NODE = False

    def encode(self, model, captions, gpus="0,1", output_dir="", overwrite=False, **kwargs):
        cfg = pack_config.load_config()
        if not output_dir:
            output_dir = cfg.get("captions_cache") or os.path.join(pack_config.PACK_DIR, "captions_cache")
        te = model.get("text_encoder") or cfg.get("text_encoder", "")
        sidecar = cfg.get("embeddings_processor_bf16", "") or \
            os.path.join(pack_config.models_root(), "diffusion_models", "embeddings_processor_bf16.safetensors")

        index_path = os.path.join(output_dir, "index.json")
        _dc = model.get("device_config") or {}
        # Collect the per-GPU layer sliders (one per detected GPU) into a list, in
        # slider order (GPU0, GPU1, ... matching the `gpus` list order).
        n_gpu = _detect_gpus()
        layers = [max(0, int(kwargs.get(f"layers_gpu{i}", 0) or 0)) for i in range(n_gpu)]
        # Keep only GPUs the user selected (len(gpus) entries) and auto-fill to 48.
        sel_gpus = [int(x) for x in str(gpus or "0,1").split(",") if x.strip() != ""] or [0, 1]
        layers = layers[:len(sel_gpus)]
        used = sum(layers)
        if used < 48:
            # fill the remainder across the selected GPUs (evenly, last-first) so the
            # split always totals 48.
            rem = 48 - used
            if layers:
                for i in range(len(layers) - 1, -1, -1):
                    add = min(rem, 48 - layers[i])
                    layers[i] += add
                    rem -= add
                    if rem <= 0:
                        break
        layers_per_gpu = ",".join(str(x) for x in layers[:len(sel_gpus)])
        if overwrite or not os.path.exists(index_path):
            rc, tail = engine_driver.run_engine("encode_captions.py", [
                "--text-encoder", te, "--sidecar", sidecar,
                "--captions", captions, "--out-dir", output_dir, "--gpus", gpus,
                "--layers-per-gpu", layers_per_gpu,
                "--connectors-device", _dc.get("connectors") or "gpu0"])
            ok = rc == 0
            print(f"[O2noorLTX25Int4EncodeCaptions] encode rc={rc} gpus={gpus} layers_per_gpu={layers_per_gpu}\n{tail}", flush=True)
        else:
            ok = True
            tail = "cache exists — reused (set overwrite to re-encode)"

        cap = {"output_dir": output_dir, "index_path": index_path, "ok": ok, "tail": tail,
               "gpus": gpus, "layers_per_gpu": layers_per_gpu}
        print(f"[O2noorLTX25Int4EncodeCaptions] done ok={ok}", flush=True)
        return (cap,)
