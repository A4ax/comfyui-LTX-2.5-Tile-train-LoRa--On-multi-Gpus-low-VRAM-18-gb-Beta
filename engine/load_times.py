"""Shared helper to record per-model load times into a JSONL file.

Each engine script (encode_captions / process_videos / precompute_audio_embeds /
train_parallel) appends a line {"component": ..., "load_s": ...} to a shared
load_times.jsonl (in the dataset root). The Metrics/Summary nodes read it back
to show how long each model took to load and the total.
"""
import json
import os


def record(path, component, load_s):
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"component": component, "load_s": round(load_s, 2)}) + "\n")
    except Exception:
        pass
