"""LTX-2.5 Int4 Tile-Train â€” ComfyUI custom node pack.

Drives the sharded + tiled int4 training engine from ComfyUI. Nodes shell out to
the engine venv (ComfyUI's own venv lacks optimum/quanto + bitsandbytes). A
single O2noorLTX25Int4LoadModel node is the source of truth (outputs LTX25_MODEL);
every other node takes that MODEL as input. Includes a custom web widget for
uploading images/voices directly into the Dataset node.
"""
import json
import os
import shutil

import folder_paths
import server
from aiohttp import web

from .nodes.load_model_node import O2noorLTX25Int4LoadModel
from .nodes.setup_node import O2noorLTX25Int4Setup
from .nodes.tile_config_node import O2noorLTX25Int4TileConfig
from .nodes.dataset_node import O2noorLTX25Int4Dataset
from .nodes.voice_dataset_node import O2noorLTX25Int4VoiceDataset
from .nodes.encode_captions_node import O2noorLTX25Int4EncodeCaptions
from .nodes.train_node import O2noorLTX25Int4Train
from .nodes.logs_node import O2noorLTX25Int4LogsOutputs
from .nodes.validate_node import O2noorLTX25Int4Validate
from .nodes.preview_dataset_node import O2noorLTX25Int4PreviewDataset
from .nodes.batch_prompt_node import O2noorLTX25BatchPrompt
from .nodes.progress_summary_node import O2noorLTX25Int4Progress, O2noorLTX25Int4SummaryViewer
from .nodes.metrics_node import O2noorLTX25Int4Metrics
from .nodes.system_monitor_node import O2noorLTX25Int4SystemMonitor
from .nodes.chunk_ffn_node import O2noorLTX25ChunkFeedForward

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "O2noorLTX25Int4LoadModel": O2noorLTX25Int4LoadModel,
    "O2noorLTX25Int4Setup": O2noorLTX25Int4Setup,
    "O2noorLTX25Int4TileConfig": O2noorLTX25Int4TileConfig,
    "O2noorLTX25Int4Dataset": O2noorLTX25Int4Dataset,
    "O2noorLTX25Int4VoiceDataset": O2noorLTX25Int4VoiceDataset,
    "O2noorLTX25Int4EncodeCaptions": O2noorLTX25Int4EncodeCaptions,
    "O2noorLTX25Int4Train": O2noorLTX25Int4Train,
    "O2noorLTX25Int4LogsOutputs": O2noorLTX25Int4LogsOutputs,
    "O2noorLTX25Int4Validate": O2noorLTX25Int4Validate,
    "O2noorLTX25Int4PreviewDataset": O2noorLTX25Int4PreviewDataset,
    "O2noorLTX25BatchPrompt": O2noorLTX25BatchPrompt,
    "O2noorLTX25Int4Progress": O2noorLTX25Int4Progress,
    "O2noorLTX25Int4SummaryViewer": O2noorLTX25Int4SummaryViewer,
    "O2noorLTX25Int4Metrics": O2noorLTX25Int4Metrics,
    "O2noorLTX25Int4SystemMonitor": O2noorLTX25Int4SystemMonitor,
    "O2noorLTX25ChunkFeedForward": O2noorLTX25ChunkFeedForward,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "O2noorLTX25Int4LoadModel": "O2noor LTX 2.5 Int4 Load Model",
    "O2noorLTX25Int4Setup": "O2noor LTX 2.5 Int4 Setup",
    "O2noorLTX25Int4TileConfig": "O2noor LTX 2.5 Int4 Tile Config",
    "O2noorLTX25Int4Dataset": "O2noor LTX 2.5 Int4 Dataset",
    "O2noorLTX25Int4VoiceDataset": "O2noor LTX 2.5 Voice Dataset (Face+Voice)",
    "O2noorLTX25Int4EncodeCaptions": "O2noor LTX 2.5 Int4 Encode Captions",
    "O2noorLTX25Int4Train": "O2noor LTX 2.5 Int4 Train",
    "O2noorLTX25Int4LogsOutputs": "O2noor LTX 2.5 Int4 Logs/Outputs",
    "O2noorLTX25Int4Validate": "O2noor LTX 2.5 Int4 Validate",
    "O2noorLTX25Int4PreviewDataset": "O2noor LTX 2.5 Int4 Preview Dataset",
    "O2noorLTX25BatchPrompt": "O2noor LTX 2.5 Batch Prompt (100 poses)",
    "O2noorLTX25Int4Progress": "O2noor LTX 2.5 Progress (Live)",
    "O2noorLTX25Int4SummaryViewer": "O2noor LTX 2.5 Summary (Info)",
    "O2noorLTX25Int4Metrics": "O2noor LTX 2.5 Metrics Dashboard",
    "O2noorLTX25Int4SystemMonitor": "O2noor LTX 2.5 System Monitor",
    "O2noorLTX25ChunkFeedForward": "modify version from kjNodes ltx 2.5 Chunk FeedForward",
}

__all__ = ["WEB_DIRECTORY", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


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


def _video_duration(video_path):
    """Probe video duration in seconds by parsing ffmpeg's `Duration:` line.
    Works with any ffmpeg (including the bundled imageio one) — no ffprobe needed."""
    try:
        import re
        import subprocess
        r = subprocess.run([_find_ffmpeg(), "-hide_banner", "-i", video_path],
                           capture_output=True, text=True, timeout=30)
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr or "")
        if m:
            h, mm, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
            return h * 3600 + mm * 60 + s
    except Exception:
        pass
    return 0.0


def _make_video_thumb(video_path, thumb_path):
    """Extract a png thumbnail at ~15% of the video (min 0.5s) so it is not a black
    lead-in frame; /view then serves it as an image."""
    try:
        import subprocess
        dur = _video_duration(video_path)
        seek = max(0.5, dur * 0.15) if dur > 0.5 else 0.5
        r = subprocess.run(
            [_find_ffmpeg(), "-y", "-ss", f"{seek:.3f}", "-i", video_path, "-frames:v", "1",
             "-vf", "scale=160:-1", "-q:v", "3", thumb_path],
            capture_output=True, text=True, timeout=60)
        return r.returncode == 0 and os.path.exists(thumb_path)
    except Exception:
        return False


# ---- media upload endpoint (images / voices -> ComfyUI input dir) ----------
async def _upload_images(request: web.Request) -> web.Response:
    from . import pack_config
    reader = await request.multipart()
    dest = pack_config.media_upload_dir()
    os.makedirs(dest, exist_ok=True)
    saved = []
    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name not in ("files", "file", "image", "video", "voice"):
            await part.read()
            continue
        is_video = part.name in ("video", "voice")
        filename = os.path.basename(part.filename or (f"clip_{len(saved)}.mp4" if is_video else f"img_{len(saved)}.png"))
        out = os.path.join(dest, filename)
        n = 1
        while os.path.exists(out):
            base, ext = os.path.splitext(filename)
            out = os.path.join(dest, f"{base}_{n}{ext}")
            n += 1
        with open(out, "wb") as fh:
            fh.write(await part.read())
        saved.append(os.path.basename(out))
        if is_video:
            # First-frame thumbnail so the frontend can show a visual preview.
            thumb = os.path.join(dest, os.path.splitext(os.path.basename(out))[0] + ".thumb.png")
            _make_video_thumb(out, thumb)
    return web.json_response({"ok": True, "files": saved})


async def _telemetry(request: web.Request) -> web.Response:
    """Return the tail of a telemetry.jsonl file (live training progress)."""
    path = request.query.get("path", "")
    if not path or not os.path.exists(path):
        return web.json_response({"ok": False, "error": "no telemetry file"})
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()[-1000:]
        return web.json_response({"ok": True, "lines": lines})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)})


async def _output_dir(request: web.Request) -> web.Response:
    """Return ComfyUI's output directory (so web widgets can re-derive a run's
    telemetry/run.json path after a workflow reload, where runtime state is lost)."""
    try:
        return web.json_response({"ok": True, "output_dir": folder_paths.get_output_directory()})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)})


async def _clear_run(request: web.Request) -> web.Response:
    """Clear a run's telemetry + run.json so the Progress/Logs/Summary widgets
    reset to a fresh 'waiting for a run' state. Accepts a telemetry path, run dir,
    or run.json path."""
    path = request.query.get("path", "")
    if not path:
        return web.json_response({"ok": False, "error": "no path"})
    try:
        tel, run_json = None, None
        if os.path.isdir(path):
            tel = os.path.join(path, "telemetry.jsonl")
            run_json = os.path.join(path, "run.json")
        elif path.endswith(".jsonl"):
            tel = path
            run_json = os.path.join(os.path.dirname(path), "run.json")
        else:  # run.json or other
            run_json = path
            tel = os.path.join(os.path.dirname(path), "telemetry.jsonl")
        if tel and os.path.exists(tel):
            with open(tel, "w", encoding="utf-8") as f:
                f.write("")
        if run_json and os.path.exists(run_json):
            os.remove(run_json)
        return web.json_response({"ok": True})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)})


async def _thumb(request: web.Request) -> web.Response:
    """On-demand first-frame thumbnail for an uploaded video. Generates + caches it
    on first request so any video (old or new) gets a preview without re-uploading."""
    from . import pack_config
    name = os.path.basename(request.query.get("file", "") or "")
    if not name:
        return web.Response(status=404)
    dest = pack_config.media_upload_dir()
    video = os.path.join(dest, name)
    if not os.path.isfile(video):
        return web.Response(status=404)
    thumb = os.path.join(dest, os.path.splitext(name)[0] + ".thumb.png")
    if not os.path.exists(thumb):
        if not _make_video_thumb(video, thumb):
            return web.Response(status=404)
    with open(thumb, "rb") as f:
        return web.Response(body=f.read(), content_type="image/png")


async def _runinfo(request: web.Request) -> web.Response:
    """Return the parsed run.json for a given path (or a run dir/telemetry path)."""
    path = request.query.get("path", "")
    if not path:
        return web.json_response({"ok": False, "error": "no path"})
    try:
        if path.lower().endswith(".jsonl"):
            path = os.path.join(os.path.dirname(path), "run.json")
        elif os.path.isdir(path):
            path = os.path.join(path, "run.json")
        if not os.path.exists(path):
            return web.json_response({"ok": False, "error": "no run.json"})
        with open(path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        return web.json_response({"ok": True, "run": data})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)})


async def _latest_run(request: web.Request) -> web.Response:
    """Find the most recent run dir under the output folder whose name is exactly
    `name` or starts with `name_` (auto_unique timestamp suffix). Returns its
    telemetry path so widgets can show the latest run after a reload."""
    name = request.query.get("name", "")
    if not name:
        return web.json_response({"ok": False, "error": "no name"})
    out = folder_paths.get_output_directory()
    matches = []
    if os.path.isdir(out):
        try:
            for d in os.listdir(out):
                full = os.path.join(out, d)
                if os.path.isdir(full) and (d == name or d.startswith(name + "_")):
                    matches.append(full)
        except Exception:
            pass
    if not matches:
        return web.json_response({"ok": False, "error": "no run"})

    def _mtime(p):
        t = os.path.join(p, "telemetry.jsonl")
        return os.path.getmtime(t) if os.path.exists(t) else 0

    best = max(matches, key=_mtime)
    return web.json_response({"ok": True, "run_dir": best,
                              "telemetry_path": os.path.join(best, "telemetry.jsonl")})


async def _system(request: web.Request) -> web.Response:
    """Live system stats: every GPU (memory used/total, util, temp) via nvidia-smi,
    plus system RAM and CPU via psutil. Queried on demand by the System Monitor
    widget."""
    gpus = []
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,"
             "utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            for line in r.stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 6:
                    continue
                def _num(s):
                    try:
                        return float(s.replace("MiB", "").replace("GB", "").replace("%", "").strip())
                    except Exception:
                        return 0.0
                gpus.append({
                    "index": int(_num(parts[0])),
                    "name": parts[1],
                    "mem_used_gb": round(_num(parts[2]) / 1024, 2),
                    "mem_total_gb": round(_num(parts[3]) / 1024, 2),
                    "util": _num(parts[4]),
                    "temp": _num(parts[5]),
                })
    except Exception:
        gpus = []

    ram = {"used_gb": 0.0, "total_gb": 0.0, "percent": 0.0}
    cpu = {"percent": 0.0, "cores_logical": 0, "cores_physical": 0}
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram = {"used_gb": round(vm.used / 1e9, 2), "total_gb": round(vm.total / 1e9, 2),
               "percent": round(vm.percent, 1)}
        cpu = {"percent": round(psutil.cpu_percent(interval=0.1), 1),
               "cores_logical": psutil.cpu_count(logical=True) or 0,
               "cores_physical": psutil.cpu_count(logical=False) or 0}
    except Exception:
        pass

    return web.json_response({"ok": True, "gpus": gpus, "ram": ram, "cpu": cpu})


async def _load_times(request: web.Request) -> web.Response:
    """Read the per-model load times (load_times.jsonl) in a dataset root."""
    path = request.query.get("path", "")
    if not path:
        return web.json_response({"ok": False, "error": "no path"})
    f = os.path.join(path, "load_times.jsonl")
    items = []
    if os.path.exists(f):
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            items.append(json.loads(line))
                        except Exception:
                            pass
        except Exception:
            pass
    return web.json_response({"ok": True, "items": items})


routes = server.PromptServer.instance.routes
routes.post("/ltx25/upload/images")(_upload_images)
routes.get("/ltx25/telemetry")(_telemetry)
routes.get("/ltx25/output_dir")(_output_dir)
routes.get("/ltx25/runinfo")(_runinfo)
routes.post("/ltx25/clear_run")(_clear_run)
routes.get("/ltx25/thumb")(_thumb)
routes.get("/ltx25/load_times")(_load_times)
routes.get("/ltx25/latest_run")(_latest_run)
routes.get("/ltx25/system")(_system)
