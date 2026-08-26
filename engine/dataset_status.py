"""Shared helper to append live dataset-pipeline status events to a JSONL file.

Each stage of the Voice Dataset pipeline (ffmpeg clip cutting, caption encoding,
audio-embed precompute, video VAE encode, audio VAE encode) appends one compact
line {"t": <epoch_s>, ...fields} to status.jsonl (in the dataset root). The
Dataset Timeline node polls this file live and renders a moving dashboard, so a
long-running encode never looks frozen.
"""
import json
import os
import time


def status(path, **fields):
    """Append a status event to `path`. Safe no-op on path/disk errors.

    `fields` may be arbitrary JSON-serializable keys; `t` (epoch seconds) and
    `ts` (clock string) are always added. Use `append=True`-style semantics:
    each call adds a newline-delimited record. Never raises.
    """
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        rec = dict(fields)
        rec["t"] = round(time.time(), 3)
        rec["ts"] = time.strftime("%H:%M:%S")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    except Exception:
        pass
