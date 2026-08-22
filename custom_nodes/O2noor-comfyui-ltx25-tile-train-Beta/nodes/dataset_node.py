"""O2noorLTX25Int4Dataset — face-only dataset builder (I2V first-frame).

Upload as many character/face IMAGES directly into the node (custom web widget).
Each image becomes a short clip whose first frame is the image (I2V first-frame
conditioning), then captions + VAE latents are computed on the chosen device.
Output carries the resolution so Train needs no duplicate resolution input.
Face-only: no voice / audio.
"""
import json
import os
import subprocess

from .. import pack_config
from . import engine_driver


def _find_ffmpeg():
    """Locate an ffmpeg binary (PATH first, then imageio_ffmpeg's bundled one)."""
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _has_nvenc():
    """True if the GPU can encode H.264 (NVENC). Cached."""
    if getattr(_has_nvenc, "_cached", None) is not None:
        return _has_nvenc._cached
    try:
        import subprocess as _sp
        r = _sp.run([_find_ffmpeg(), "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30)
        _has_nvenc._cached = "h264_nvenc" in (r.stdout + r.stderr)
    except Exception:
        _has_nvenc._cached = False
    return _has_nvenc._cached


def _dev_choice(choice, default="cuda:0"):
    """Map a device-config value ('gpu0'..'gpu3'|'cpu') to a torch device string."""
    return {"gpu0": "cuda:0", "gpu1": "cuda:1", "gpu2": "cuda:2", "gpu3": "cuda:3",
            "cpu": "cpu"}.get(choice, default)


def _gemma_device(device, fallback="0"):
    """Map the Gemma device choice ('gpu0'|'gpu1'|'gpu2') to a GPU index for --gpus."""
    return {"gpu0": "0", "gpu1": "1", "gpu2": "2"}.get(device, fallback)


class O2noorLTX25Int4Dataset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LTX25_MODEL", {"tooltip": "The int4 model from O2noorLTX25Int4LoadModel."}),
                "images": ("STRING", {
                    "default": "",
                    "tooltip": "Upload your character/face images here (add as many as you want).",
                }),
                "width": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 64,
                    "tooltip": "Output width (pixels). Must be a multiple of 32.",
                }),
                "height": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 64,
                    "tooltip": "Output height (pixels). Must be a multiple of 32.",
                }),
                "frames": ("INT", {
                    "default": 17, "min": 9, "max": 129, "step": 8,
                    "tooltip": "Frames per training clip (17 = ~0.7s at 24fps). Must be (8k+1).",
                }),
                "clip_fps": ("INT", {
                    "default": 24, "min": 8, "max": 60, "step": 1,
                    "tooltip": "Frames-per-second of the generated training clips.",
                }),
                "trigger_word": ("STRING", {
                    "default": "ltxchar",
                    "tooltip": "Trigger word prepended to every caption (the LoRA activates on this word).",
                }),
                "device": (engine_driver.DEVICE_CHOICES, {
                    "default": "auto",
                    "tooltip": "Where VAE encoding runs: auto = first free GPU, else cpu.",
                }),
                "output_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Where latents/conditions are written. Empty = the pack's dataset folder (config.json).",
                }),
                "vae_tiling": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Tile the VAE for larger resolutions (avoids OOM).",
                }),
            },
            "optional": {
                "captions": ("LTX25_CAPTIONS", {
                    "tooltip": "Optional pre-encoded captions from O2noorLTX25Int4EncodeCaptions. "
                               "If omitted, captions are encoded automatically on the GPUs.",
                }),
            }
        }

    RETURN_TYPES = ("LTX25_DATASET",)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "preprocess"
    CATEGORY = "ltx25-int4-train"
    TITLE = "O2noor LTX 2.5 Int4 Dataset"

    def preprocess(self, model, images, width=512, height=512,
                   frames=17, clip_fps=24, trigger_word="ltxchar", device="auto",
                   output_dir="", vae_tiling=True, captions=None):
        cfg = pack_config.load_config()
        if not output_dir:
            output_dir = cfg.get("dataset_root") or os.path.join(pack_config.PACK_DIR, "dataset")

        # GPU-encoded captions: use the EncodeCaptions output if wired in, otherwise
        # run the 8-bit Gemma on the GPUs automatically (never CPU).
        cap_src = captions.get("output_dir") if captions else None
        if cap_src and not os.path.exists(os.path.join(cap_src, "index.json")):
            cap_src = None
        dev = engine_driver.pick_device(device)
        print(f"[O2noorLTX25Int4Dataset] device resolved: {dev}", flush=True)

        def parse_list(raw):
            raw = (raw or "").strip()
            if not raw:
                return []
            try:
                v = json.loads(raw)
                return v if isinstance(v, list) else [raw]
            except Exception:
                return [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]

        image_paths = parse_list(images)

        if not image_paths:
            return ({"dataset_root": output_dir, "ok": False,
                     "log_tail": "no images uploaded — add character images to the node"},)

        media_root = pack_config.media_upload_dir()
        # resolve upload paths (they're relative to input dir)
        def resolve(p):
            full = os.path.join(media_root, p) if not os.path.isabs(p) else p
            return full if os.path.exists(full) else p

        image_paths = [resolve(p) for p in image_paths]
        print(f"[O2noorLTX25Int4Dataset] {len(image_paths)} images", flush=True)

        os.makedirs(output_dir, exist_ok=True)
        clips_dir = os.path.join(output_dir, "scenes")
        os.makedirs(clips_dir, exist_ok=True)
        log_parts = []

        # 1) each image -> a short clip (first frame = the image) via ffmpeg
        clips = []
        ffmpeg = _find_ffmpeg()
        nvenc = _has_nvenc()
        for i, img in enumerate(image_paths):
            out = os.path.join(clips_dir, f"img_{i:03d}.mp4")
            dur = frames / max(1, clip_fps)
            cmd = [ffmpeg, "-y", "-loop", "1", "-i", img, "-t", f"{dur:.2f}",
                   "-r", str(clip_fps)]
            cmd += ["-vf", f"scale={width}:{height}", "-c:v", ("h264_nvenc" if nvenc else "libx264")]
            cmd += ["-pix_fmt", "yuv420p", out]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if r.returncode == 0 and os.path.exists(out):
                clips.append(os.path.join("scenes", os.path.basename(out)))
            else:
                log_parts.append(f"[img {i}] ffmpeg failed: {(r.stderr or '')[-400:]}")
        if not clips:
            return ({"dataset_root": output_dir, "ok": False, "log_tail": "\n".join(log_parts) or "ffmpeg failed"},)

        # 2) build dataset.json (I2V: video column; caption = trigger)
        samples = [{"video": c, "caption": trigger_word} for c in clips]
        dataset_json = os.path.join(output_dir, "dataset.json")
        with open(dataset_json, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)
        log_parts.append(f"dataset.json written ({len(samples)} samples)")

        # 3) captions -> conditions from the GPU-encoded embeddings (never CPU).
        #    Use the EncodeCaptions cache if wired in; otherwise run the 8-bit Gemma
        #    on the GPUs automatically.
        cap_out = os.path.join(output_dir, "conditions")
        os.makedirs(cap_out, exist_ok=True)
        _dc = model.get("device_config") or {}
        _tec = _dc.get("text_encoder") or {}
        _te_gpus = [_gemma_device(_tec.get("device"))]
        if cap_src is None:
            cap_src = os.path.join(output_dir, "captions_cache")
            rc, tail = engine_driver.run_engine("encode_captions.py", [
                "--text-encoder", model.get("text_encoder") or cfg.get("text_encoder", ""),
                "--sidecar", cfg.get("embeddings_processor_bf16", ""),
                "--captions", trigger_word,
                "--out-dir", cap_src,
                "--gpus", ",".join(_te_gpus),
                "--connectors-device", _dc.get("connectors") or "gpu0"])
            log_parts.append(f"[captions] auto GPU encode rc={rc}\n{tail}")
            if rc != 0:
                return ({"dataset_root": output_dir, "ok": False, "log_tail": "\n".join(log_parts)},)
        with open(os.path.join(cap_src, "index.json"), encoding="utf-8") as f:
            cap_index = json.load(f)
        by_caption = {e.get("caption"): e for e in cap_index}
        entry = by_caption.get(trigger_word) or (cap_index[0] if cap_index else None)
        cond_file = (entry or {}).get("path") or (entry or {}).get("cond")
        if entry is None or not os.path.exists(cond_file or ""):
            return ({"dataset_root": output_dir, "ok": False,
                     "log_tail": f"no cached condition found for caption '{trigger_word}' in {cap_src}. "
                                 "Run Encode Captions with that caption."},)
        import shutil
        for s in samples:
            video_rel = s["video"]  # e.g. scenes/img_000.mp4
            out_cond = os.path.join(cap_out, os.path.splitext(video_rel)[0] + ".pt")
            os.makedirs(os.path.dirname(out_cond), exist_ok=True)
            shutil.copyfile(cond_file, out_cond)
        log_parts.append(f"[captions] copied GPU-encoded conditions (caption '{trigger_word}')")

        # 4) VAE latents (video) on the chosen device
        model_path = model.get("int4_model_file")
        vae_dev = _dev_choice(_dc.get("video_vae"), dev)
        vid_args = [dataset_json, "--resolution-buckets", f"{width}x{height}x{frames}",
                    "--output-dir", os.path.join(output_dir, "latents"),
                    "--model-path", model_path, "--device", vae_dev,
                    "--video-column", "video"]
        vvae = model.get("video_vae")
        if vvae and os.path.exists(vvae):
            vid_args += ["--video-vae-path", vvae]
        if vae_tiling:
            vid_args.append("--vae-tiling")
        vid_args.append("--overwrite")  # regenerate latents so a fresh upload always encodes
        rc, tail = engine_driver.run_engine("process_videos.py", vid_args)
        log_parts.append(f"[process_videos] rc={rc}\n{tail}")
        if rc != 0:
            return ({"dataset_root": output_dir, "ok": False, "log_tail": "\n".join(log_parts)},)

        ds = {
            "dataset_root": output_dir,
            "dataset_json": dataset_json,
            "width": width, "height": height, "frames": frames,
            "device": dev,
            "ok": True,
            "samples": len(samples),
            "images": image_paths,  # resolved training images (for the Preview node)
            "log_tail": "\n".join(log_parts),
        }
        print(f"[O2noorLTX25Int4Dataset] done ok=True device={dev} samples={len(samples)}", flush=True)
        return (ds,)
