"""O2noorLTX25Int4LoadModel — single model source of truth + device placement (face-only).

Dropdowns for the int4 model, connectors, text encoder and video VAE, PLUS per-component
device placement:
  - Transformer: BLOCKS per GPU (how many of the 48 blocks go to each GPU).
  - Gemma text encoder: a GPU choice (dropdown) — runs 8-bit (~8 GB) on that card.
  - Connectors / video VAE: a device choice.
Outputs an LTX25_MODEL bundle (paths + device_config). Face-only: no audio VAE.
"""
import os

import folder_paths

from .. import pack_config
from . import engine_driver


def _dropdown(folder, extra_path=None, default_name=None):
    files = list(folder_paths.get_filename_list(folder))
    if extra_path and os.path.exists(extra_path) and os.path.basename(extra_path) not in files:
        files.append(os.path.basename(extra_path))
    files = sorted(set(files))
    default = default_name or (files[0] if files else "NONE")
    if default not in files and files:
        default = files[0]
    return files, default


def _detect_gpus():
    """Number of physical CUDA GPUs (independent of ComfyUI's in-process mask)."""
    return engine_driver.detect_cuda_gpus()


def _gpu_names(n):
    return [f"gpu{i}" for i in range(n)]


def _device_choices(n=None, with_cpu=False):
    n = n if n is not None else _detect_gpus()
    names = _gpu_names(n)
    return names + (["cpu"] if with_cpu else [])


def _default_blocks(n):
    """Default transformer block distribution (Option B/B2):
    1 GPU -> [48]; 2 GPUs -> [24, 24]; 3+ GPUs -> fill first n-1 evenly, last = 0."""
    if n <= 1:
        return [48]
    if n == 2:
        return [24, 24]
    used = n - 1
    base = 48 // used
    rem = 48 % used
    blocks = [base + (1 if i < rem else 0) for i in range(used)] + [0]
    return blocks


class O2noorLTX25Int4LoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        cfg = pack_config.load_config()
        int4_files, int4_default = _dropdown(
            "diffusion_models",
            extra_path=cfg.get("int4_model_file", ""),
            default_name="ltx-2.5-22b-distilled-int4-main-v2.safetensors")
        conn_files, conn_default = _dropdown(
            "diffusion_models",
            extra_path=cfg.get("connectors_bf16", ""))
        te_files, te_default = _dropdown(
            "text_encoders",
            extra_path=cfg.get("text_encoder", ""),
            default_name="gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
        vvae_files, vvae_default = _dropdown(
            "vae",
            extra_path=cfg.get("video_vae", ""))
        avvae_files, avvae_default = _dropdown(
            "vae",
            extra_path=cfg.get("audio_vae", ""))
        blk = {"min": 0, "max": 48, "step": 1}
        n_gpu = _detect_gpus()
        dev = _device_choices(n_gpu, with_cpu=True)
        required = {
            "int4_model": (int4_files, {
                "default": int4_default,
                "tooltip": "The pre-built int4 model (.safetensors). Loads into GPU(s), block by block.",
            }),
            "connectors": (conn_files, {
                "default": conn_default,
                "tooltip": "connectors_bf16.safetensors — text-embedding connectors for caption conditioning.",
            }),
            "text_encoder": (te_files, {
                "default": te_default,
                "tooltip": "LTX-2.5 Gemma-4 text encoder (use the bf16 build; the engine 8-bit loads it across GPUs).",
            }),
            "video_vae": (vvae_files, {
                "default": vvae_default,
                "tooltip": "LTX-2.5 video VAE (encodes your face images into latents).",
            }),
            "audio_vae": (avvae_files, {
                "default": avvae_default,
                "tooltip": "LTX-2.5 audio VAE (encodes the voice track into audio latents for voice+face training).",
            }),
        }
        # ---- Transformer: BLOCKS per GPU (48 blocks total), one per detected GPU ----
        defaults = _default_blocks(n_gpu)
        for i in range(n_gpu):
            required[f"transformer_blocks_gpu{i}"] = ("INT", {**blk, "default": defaults[i],
                "tooltip": f"How many of the 48 transformer blocks go to GPU {i}. 0 = don't use GPU {i}."})
        # ---- Connectors / video VAE devices ----
        required["connectors_device"] = (dev, {"default": "gpu0",
            "tooltip": "Which device runs the connectors (embeddings processor)."})
        required["video_vae_device"] = (dev, {"default": f"gpu{n_gpu - 1}",
            "tooltip": "Which device runs the video VAE."})
        required["audio_vae_device"] = (dev, {"default": f"gpu{n_gpu - 1}",
            "tooltip": "Which device runs the audio VAE (voice encoding)."})
        return {"required": required}

    RETURN_TYPES = ("LTX25_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "ltx25-int4-train"
    TITLE = "O2noor LTX 2.5 Int4 Load Model"

    def load(self, int4_model, connectors, text_encoder, video_vae, audio_vae,
             connectors_device, video_vae_device, audio_vae_device, **kwargs):
        def path_of(folder, name):
            try:
                return folder_paths.get_full_path(folder, name)
            except Exception:
                return name

        cfg = pack_config.load_config()
        n_gpu = _detect_gpus()
        blocks = {}
        for i in range(n_gpu):
            blocks[f"blocks_gpu{i}"] = int(kwargs.get(f"transformer_blocks_gpu{i}") or 0)
        # Clamp device selections to GPUs that actually exist on THIS machine (a saved
        # workflow may name gpu2 on a 2-GPU rig). Never pass an out-of-range index.
        def _clamp(dev, fallback):
            dev = str(dev or "")
            if dev.startswith("gpu") and dev[3:].isdigit():
                return dev if int(dev[3:]) < n_gpu else f"gpu{max(0, n_gpu - 1)}"
            return fallback if n_gpu == 0 else f"gpu{max(0, n_gpu - 1)}"
        device_config = {
            "transformer": blocks,
            "connectors": _clamp(connectors_device, "gpu0"),
            "video_vae": _clamp(video_vae_device, f"gpu{max(0, n_gpu - 1)}"),
            "audio_vae": _clamp(audio_vae_device, f"gpu{max(0, n_gpu - 1)}"),
        }
        bundle = {
            "int4_model_file": path_of("diffusion_models", int4_model),
            "connectors_bf16": path_of("diffusion_models", connectors),
            "embeddings_processor_bf16": cfg.get("embeddings_processor_bf16", ""),
            "text_encoder": path_of("text_encoders", text_encoder),
            "video_vae": path_of("vae", video_vae),
            "audio_vae": path_of("vae", audio_vae),
            "engine_python": cfg.get("engine_python", ""),
            "engine_workdir": cfg.get("engine_workdir", ""),
            "device_config": device_config,
            "ok": True,
        }
        print(f"[O2noorLTX25Int4LoadModel] int4={int4_model} connectors={connectors} te={text_encoder}", flush=True)
        print(f"[O2noorLTX25Int4LoadModel] transformer blocks={device_config['transformer']} "
              f"connectors={connectors_device} vvae={video_vae_device} avae={audio_vae_device}", flush=True)
        return (bundle,)
