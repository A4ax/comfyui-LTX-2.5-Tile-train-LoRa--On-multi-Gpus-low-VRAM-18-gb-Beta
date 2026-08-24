"""O2noorLTX25Int4VoiceDataset — voice+face dataset builder (new engine node).

Upload VIDEOS of the person SPEAKING directly into the node. Each clip is
normalized (scale + fps + duration, audio preserved at 16 kHz) and then:
  - video VAE -> latents/            (learns the FACE)
  - audio VAE -> audio_latents/      (learns the VOICE, when mode is "face+voice")
  - captions  -> conditions/         (video + audio prompt embeds)

Mode "face+voice" trains a LoRA that captures both the face and the voice;
mode "face-only" is the existing behavior (voice ignored). Everything is
configurable (paths from pack_config), and all heavy work runs on the GPUs
selected on the O2noorLTX25Int4LoadModel node (device placement / block split).
"""
import json
import os
import subprocess

from .. import pack_config
from . import engine_driver


def _find_ffmpeg():
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
    return {"gpu0": "cuda:0", "gpu1": "cuda:1", "gpu2": "cuda:2", "gpu3": "cuda:3",
            "cpu": "cpu"}.get(choice, default)


def _gemma_device(device, fallback="0"):
    return {"gpu0": "0", "gpu1": "1", "gpu2": "2"}.get(device, fallback)


def _auto_gpus():
    """List all available CUDA devices as a comma string (never hardcode 0,1)."""
    try:
        import torch as _t
        n = _t.cuda.device_count() if _t.cuda.is_available() else 0
        return ",".join(str(i) for i in range(max(1, n)))
    except Exception:
        return "0"


def _video_duration(path):
    """Probe video duration in seconds by parsing ffmpeg's `Duration:` line."""
    try:
        import re
        import subprocess
        r = subprocess.run([_find_ffmpeg(), "-hide_banner", "-i", path],
                           capture_output=True, text=True, timeout=30)
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr or "")
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    except Exception:
        pass
    return 0.0


class O2noorLTX25Int4VoiceDataset:
    @classmethod
    def INPUT_TYPES(cls):
        cfg = pack_config.load_config()
        default_out = cfg.get("dataset_root") or os.path.join(engine_driver.engine_workdir(), "dataset")
        return {
            "required": {
                "model": ("LTX25_MODEL", {"tooltip": "The int4 model from O2noorLTX25Int4LoadModel."}),
                "videos": ("STRING", {
                    "default": "",
                    "tooltip": "Upload VIDEOS of the person speaking here. Each video is auto-split "
                               "into segment_duration clips (face+voice).",
                }),
                "images": ("STRING", {
                    "default": "",
                    "tooltip": "Optional: upload face IMAGES too. Each becomes one face-only clip "
                               "to reinforce identity (no voice).",
                }),
                "mode": (["face+voice", "face-only"], {
                    "default": "face+voice",
                    "tooltip": "face+voice trains a LoRA that learns both the face and the voice. "
                               "face-only is the original face-only behavior.",
                }),
                "segment_duration": ("FLOAT", {
                    "default": 0.7, "min": 0.2, "max": 10.0, "step": 0.1,
                    "tooltip": "Seconds per training clip. Each uploaded video is auto-split into "
                               "clips of this length (frames auto-derived to a valid 8k+1 bucket). "
                               "e.g. 0.7 = ~17 frames, 1.0 = ~25 frames at 24fps.",
                }),
                "max_segments": ("INT", {
                    "default": 0, "min": 0, "max": 500, "step": 1,
                    "tooltip": "Max clips to cut from EACH uploaded video. 0 = unlimited (cut them all).",
                }),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64,
                                  "tooltip": "Output width (pixels). Multiple of 32."}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64,
                                   "tooltip": "Output height (pixels). Multiple of 32."}),
                "clip_fps": ("INT", {"default": 24, "min": 8, "max": 60, "step": 1,
                                     "tooltip": "Frames-per-second of the generated training clips."}),
                "trigger_word": ("STRING", {"default": "ltxchar",
                    "tooltip": "Trigger word prepended to every caption (the LoRA activates on this word)."}),
                "output_dir": ("STRING", {"default": default_out,
                    "tooltip": "Where latents/audio_latents/conditions are written (any disk path)."}),
                "vae_tiling": ("BOOLEAN", {"default": True,
                    "tooltip": "Tile the VAE for larger resolutions (avoids OOM)."}),
                "vae_tile_size": ("INT", {"default": 0, "min": 0, "max": 1280, "step": 32,
                    "tooltip": "VAE tile size in pixels. 0 = auto (engine default ~512). "
                               "Slide up for more VRAM per tile, fewer tiles, faster encode. "
                               "Slide down for less VRAM, more tiles, slower."}),
                "vae_tile_overlap": ("INT", {"default": 128, "min": 0, "max": 256, "step": 32,
                    "tooltip": "VAE tile overlap in pixels (0-256). Higher = smoother seams but "
                               "more redundant compute (slower). 128 = default."}),
                "overwrite": ("BOOLEAN", {"default": False,
                    "tooltip": "Re-encode video/audio latents even if they already exist. "
                               "OFF = reuse existing latents (faster on repeated runs). "
                               "ON = force re-encode after you change the dataset."}),
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
    TITLE = "O2noor LTX 2.5 Int4 Voice Dataset"

    def preprocess(self, model, videos, images, mode="face+voice", segment_duration=0.7,
                   max_segments=0, width=512, height=512, clip_fps=24, trigger_word="ltxchar",
                   output_dir=None, vae_tiling=True, vae_tile_size=512, vae_tile_overlap=128,
                   overwrite=False, captions=None):
        cfg = pack_config.load_config()
        if not output_dir:
            output_dir = cfg.get("dataset_root") or os.path.join(engine_driver.engine_workdir(), "dataset")
        use_audio = mode == "face+voice"

        # Audio VAE comes from the LoadModel node (single source of truth), else config.
        avae = model.get("audio_vae") or cfg.get("audio_vae", "")

        # Derive the video frames bucket from the requested segment duration (snap to 8k+1).
        fps = max(1, int(clip_fps))
        target = max(1, int(round(float(segment_duration) * fps)))
        k = max(1, int(round((target - 1) / 8.0)))
        frames = 8 * k + 1
        segdur = frames / fps

        cap_src = captions.get("output_dir") if captions else None
        if cap_src and not os.path.exists(os.path.join(cap_src, "index.json")):
            cap_src = None
        # Text-encoder GPUs flow from the O2noorLTX25Int4EncodeCaptions node when wired;
        # otherwise auto-detect all available GPUs (never hardcode to GPU0/GPU1).
        te_gpus = (captions or {}).get("gpus") or _auto_gpus()
        _dc = model.get("device_config") or {}
        # Device placement comes from the LoadModel node (video_vae/audio_vae); fall back to auto.
        dev = _dev_choice(_dc.get("video_vae"), engine_driver.pick_device("auto"))
        print(f"[O2noorLTX25Int4VoiceDataset] device={dev} mode={mode} seg={segment_duration}s->{frames}f "
              f"({segdur:.3f}s) audio={use_audio}", flush=True)

        def parse_list(raw):
            raw = (raw or "").strip()
            if not raw:
                return []
            try:
                v = json.loads(raw)
                return v if isinstance(v, list) else [raw]
            except Exception:
                return [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]

        video_paths = parse_list(videos)
        image_paths = parse_list(images)
        if not video_paths and not image_paths:
            return ({"dataset_root": output_dir, "ok": False,
                     "log_tail": "no videos or images uploaded — add videos of the person speaking "
                                 "and/or face images to the node"},)

        media_root = pack_config.media_upload_dir()

        def resolve(p):
            full = os.path.join(media_root, p) if not os.path.isabs(p) else p
            return full if os.path.exists(full) else p

        video_paths = [resolve(p) for p in video_paths]
        image_paths = [resolve(p) for p in image_paths]
        print(f"[O2noorLTX25Int4VoiceDataset] {len(video_paths)} videos + {len(image_paths)} images", flush=True)

        os.makedirs(output_dir, exist_ok=True)
        clips_dir = os.path.join(output_dir, "scenes")
        os.makedirs(clips_dir, exist_ok=True)
        log_parts = []

        # Clear stale artifacts from earlier runs (different buckets) so the trainer
        # never samples leftover latents of a mismatched resolution.
        for stale in ("scenes", "latents", "audio_latents", "conditions"):
            _sd = os.path.join(output_dir, stale)
            if os.path.isdir(_sd):
                for _f in os.listdir(_sd):
                    try:
                        os.remove(os.path.join(_sd, _f))
                    except Exception:
                        pass
        log_parts.append("[clean] cleared stale scenes/latents/audio_latents/conditions")

        # Shared per-model load-time log (truncated each run); each engine script appends to it.
        lt_path = os.path.join(output_dir, "load_times.jsonl")
        with open(lt_path, "w", encoding="utf-8") as _lf:
            _lf.write("")

        # 1) cut training clips: videos -> auto-split into segdur clips (audio kept in voice mode);
        #    images -> one clip each (face-only, no audio).
        clips = []
        ffmpeg = _find_ffmpeg()
        nvenc = _has_nvenc()
        idx = 0
        n_video_clips = 0

        def make_clip(cmd, out, tag):
            nonlocal idx
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            except Exception as e:
                log_parts.append(f"[{tag}] ffmpeg error: {e}")
                return False
            ok = r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 0
            if ok:
                clips.append(os.path.join("scenes", os.path.basename(out)))
                idx += 1
                return True
            # failed or empty (e.g. seek past end of video) -> clean up + signal stop.
            if os.path.exists(out):
                try:
                    os.remove(out)
                except Exception:
                    pass
            log_parts.append(f"[{tag}] ffmpeg {'failed' if r.returncode else 'empty'}: {(r.stderr or '')[-300:]}")
            return False

        for v in video_paths:
            # ffmpeg -ss before -i does fast input seek; each segment is segdur long.
            # Stop when we hit max_segments, run out of video, or produce an empty segment.
            dur = _video_duration(v)
            cap = max_segments if max_segments and max_segments > 0 else None
            seg = 0
            while True:
                start = seg * segdur
                if cap is not None and seg >= cap:
                    break
                # Stop at the end of the video AND drop a partial trailing segment
                # (a clip shorter than segdur would be an incomplete 17-frame bucket).
                if dur > 0 and (start >= dur - 0.01 or start + segdur > dur + 0.01):
                    break
                out = os.path.join(clips_dir, f"img_{idx:03d}.mp4")
                cmd = [ffmpeg, "-y", "-ss", f"{start:.3f}", "-i", v, "-t", f"{segdur:.3f}",
                       "-r", str(fps), "-vf", f"scale={width}:{height}"]
                cmd += ["-c:v", ("h264_nvenc" if nvenc else "libx264")]
                if use_audio:
                    cmd += ["-c:a", "aac", "-ac", "2", "-ar", "16000"]
                else:
                    cmd += ["-an"]
                cmd += ["-pix_fmt", "yuv420p", out]
                if not make_clip(cmd, out, f"vid-seg {seg}"):
                    break
                n_video_clips += 1
                seg += 1
            log_parts.append(f"[split] video split into {seg} x {segdur:.3f}s clips (dur={dur:.1f}s)")

        for i, img in enumerate(image_paths):
            out = os.path.join(clips_dir, f"img_{idx:03d}.mp4")
            cmd = [ffmpeg, "-y", "-loop", "1", "-i", img, "-t", f"{segdur:.3f}", "-r", str(fps),
                   "-vf", f"scale={width}:{height}"]
            cmd += ["-c:v", ("h264_nvenc" if nvenc else "libx264"), "-an", "-pix_fmt", "yuv420p", out]
            make_clip(cmd, out, f"img {i}")

        if not clips:
            return ({"dataset_root": output_dir, "ok": False, "log_tail": "\n".join(log_parts) or "ffmpeg failed"},)

        # 1b) Extract the voice track from each VIDEO clip into a .wav so process_videos
        #     --with-audio has an explicit `audio` column to encode into audio_latents.
        audio_rels = [None] * len(clips)
        if use_audio and n_video_clips > 0:
            audio_dir = os.path.join(output_dir, "audio")
            os.makedirs(audio_dir, exist_ok=True)
            for i in range(n_video_clips):
                clip_abs = os.path.join(output_dir, clips[i])   # clips[i] = "scenes/img_XXX.mp4"
                wav_name = os.path.splitext(os.path.basename(clip_abs))[0] + ".wav"
                wav_abs = os.path.join(audio_dir, wav_name)
                r = subprocess.run([ffmpeg, "-y", "-i", clip_abs, "-vn", "-ac", "1", "-ar", "16000",
                                    "-c:a", "pcm_s16le", wav_abs],
                                   capture_output=True, text=True, timeout=120)
                if r.returncode == 0 and os.path.exists(wav_abs) and os.path.getsize(wav_abs) > 0:
                    audio_rels[i] = os.path.join("audio", wav_name)
                else:
                    log_parts.append(f"[audio] extract failed for {clips[i]}: {(r.stderr or '')[-300:]}")
            log_parts.append(f"[audio] extracted voice for {sum(1 for a in audio_rels if a)}/{n_video_clips} video clips")

        # 2) build dataset.json (audio column only for video clips, in voice mode)
        samples = []
        for i, c in enumerate(clips):
            entry = {"video": c, "caption": trigger_word}
            if use_audio and i < n_video_clips and audio_rels[i]:
                entry["audio"] = audio_rels[i]
            samples.append(entry)
        dataset_json = os.path.join(output_dir, "dataset.json")
        with open(dataset_json, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)
        log_parts.append(f"dataset.json written ({len(samples)} samples, mode={mode})")

        # 3) captions -> conditions (video + audio prompt embeds) on the GPUs.
        cap_out = os.path.join(output_dir, "conditions")
        os.makedirs(cap_out, exist_ok=True)
        if cap_src is None:
            cap_src = os.path.join(output_dir, "captions_cache")
            rc, tail = engine_driver.run_engine("encode_captions.py", [
                "--text-encoder", model.get("text_encoder") or cfg.get("text_encoder", ""),
                "--sidecar", cfg.get("embeddings_processor_bf16", ""),
                "--captions", trigger_word,
                "--out-dir", cap_src,
                "--gpus", te_gpus,
                "--connectors-device", _dc.get("connectors") or "gpu0",
                "--load-times", lt_path])
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
                     "log_tail": f"no cached condition found for caption '{trigger_word}' in {cap_src}."},)
        import shutil
        for s in samples:
            video_rel = s["video"]
            out_cond = os.path.join(cap_out, os.path.splitext(video_rel)[0] + ".pt")
            os.makedirs(os.path.dirname(out_cond), exist_ok=True)
            shutil.copyfile(cond_file, out_cond)
        log_parts.append(f"[captions] copied GPU-encoded conditions (caption '{trigger_word}')")

        # 3b) In voice mode, precompute the connector-applied video/audio embeds OFFLINE
        #     (once) so the training engine does NOT load the embeddings processor on the
        #     GPUs -> keeps training VRAM at face-only levels (no OOM).
        if use_audio:
            rc, tail = engine_driver.run_engine("precompute_audio_embeds.py", [
                "--conditions-dir", cap_out,
                "--sidecar", cfg.get("embeddings_processor_bf16", ""),
                "--text-encoder", model.get("text_encoder") or cfg.get("text_encoder", ""),
                "--gpus", te_gpus,
                "--load-times", lt_path])
            log_parts.append(f"[audio] precompute embeds rc={rc}\n{tail}")
            if rc != 0:
                return ({"dataset_root": output_dir, "ok": False, "log_tail": "\n".join(log_parts)},)

        # 4) video VAE latents (face) on the chosen device.
        model_path = model.get("int4_model_file")
        vae_dev = _dev_choice(_dc.get("video_vae"), dev)
        vid_args = [dataset_json, "--resolution-buckets", f"{width}x{height}x{frames}",
                    "--output-dir", os.path.join(output_dir, "latents"),
                    "--model-path", model_path, "--video-column", "video"]
        vvae = model.get("video_vae")
        if vvae and os.path.exists(vvae):
            vid_args += ["--video-vae-path", vvae]
        # 5) audio VAE latents (voice) — same process_videos call, when in voice mode.
        if use_audio and avae and os.path.exists(avae):
            proc_dev = _dev_choice(_dc.get("audio_vae"), vae_dev)
            vid_args += ["--with-audio", "--audio-vae-path", avae,
                         "--audio-output-dir", os.path.join(output_dir, "audio_latents")]
            log_parts.append(f"[audio] voice mode: encoding voice via audio VAE on {proc_dev}")
        else:
            proc_dev = vae_dev
            log_parts.append(f"[audio] mode={mode}: voice encoding {'skipped (no audio_vae)' if use_audio else 'off'}")
        vid_args += ["--device", proc_dev, "--load-times-path", lt_path]
        if vae_tiling:
            vid_args.append("--vae-tiling")
            if int(vae_tile_size) > 0:
                vid_args += ["--tile-size", str(int(vae_tile_size))]
            vid_args += ["--tile-overlap", str(int(vae_tile_overlap))]
        if overwrite:
            vid_args.append("--overwrite")
        rc, tail = engine_driver.run_engine("process_videos.py", vid_args)
        log_parts.append(f"[process_videos] rc={rc}\n{tail}")
        if rc != 0:
            return ({"dataset_root": output_dir, "ok": False, "log_tail": "\n".join(log_parts)},)

        ds = {
            "dataset_root": output_dir,
            "dataset_json": dataset_json,
            "width": width, "height": height, "frames": frames,
            "device": dev,
            "audio_vae": avae,
            "mode": mode,
            "use_audio": use_audio,
            "ok": True,
            "samples": len(samples),
            "videos": video_paths,
            "images": image_paths,
            "segment_duration": segdur,
            "max_segments": max_segments,
            "log_tail": "\n".join(log_parts),
        }
        print(f"[O2noorLTX25Int4VoiceDataset] done ok=True device={dev} mode={mode} "
              f"samples={len(samples)}", flush=True)
        return (ds,)
