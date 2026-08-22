"""Build the 7-node wired workflow (all visible, all connected)."""
import json
import os
import uuid

import engine_env

PACK_WF = os.path.join(engine_env.PACK_ROOT, "workflow")
USER_WF = os.environ.get("LTX_COMFY_USER_WF", "")
DATASET_DIR = engine_env.DATASET_ROOT
CAPTIONS_DIR = os.path.join(engine_env.PACK_ROOT, "captions_cache")

L = []
def link(i, s, d, di):
    L.append([i, s, 0, d, di])
    return i

nodes = []

# 1 LoadModel
nodes.append({
    "id": 1, "type": "O2noorLTX25Int4LoadModel", "pos": [30, 300], "size": [420, 320], "flags": {}, "order": 0, "mode": 0,
    "inputs": [
        {"name": "int4_model", "type": "COMBO", "widget": {"name": "int4_model"}, "link": None},
        {"name": "connectors", "type": "COMBO", "widget": {"name": "connectors"}, "link": None},
        {"name": "text_encoder", "type": "COMBO", "widget": {"name": "text_encoder"}, "link": None},
        {"name": "video_vae", "type": "COMBO", "widget": {"name": "video_vae"}, "link": None},
        {"name": "audio_vae", "type": "COMBO", "widget": {"name": "audio_vae"}, "link": None},
    ],
    "outputs": [{"name": "model", "type": "LTX25_MODEL", "links": [1, 2, 3, 4]}],
    "properties": {"Node name for S&R": "O2noorLTX25Int4LoadModel"},
    "widgets_values": ["ltx-2.5-22b-distilled-bnb-nf4.safetensors",
                       "connectors_bf16.safetensors",
                       "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
                       "ltx-2.5-video-vae-bf16.safetensors",
                       "ltx-2.5-audio-vae-bf16.safetensors"],
    "title": "LTX 2.5 Int4 Load Model",
})
link(1, 1, 2, 0)  # LoadModel -> TileConfig
link(2, 1, 3, 0)  # LoadModel -> EncodeCaptions
link(3, 1, 4, 0)  # LoadModel -> Dataset
link(4, 1, 5, 0)  # LoadModel -> Train

# 2 TileConfig
nodes.append({
    "id": 2, "type": "O2noorLTX25Int4TileConfig", "pos": [500, 40], "size": [400, 260], "flags": {}, "order": 1, "mode": 0,
    "inputs": [{"name": "model", "type": "LTX25_MODEL", "link": 1}],
    "outputs": [{"name": "tile_config", "type": "LTX25_TILECONFIG", "links": [5]}],
    "properties": {"Node name for S&R": "O2noorLTX25Int4TileConfig"},
    "widgets_values": [1, 1, 0, ""],
    "title": "LTX 2.5 Int4 Tile Config",
})
link(5, 2, 5, 2)  # TileConfig -> Train.tile_config

# 3 EncodeCaptions
nodes.append({
    "id": 3, "type": "O2noorLTX25Int4EncodeCaptions", "pos": [500, 320], "size": [400, 240], "flags": {}, "order": 2, "mode": 0,
    "inputs": [
        {"name": "model", "type": "LTX25_MODEL", "link": 2},
        {"name": "captions", "type": "STRING", "widget": {"name": "captions"}, "link": None},
        {"name": "gpus", "type": "STRING", "widget": {"name": "gpus"}, "link": None},
        {"name": "output_dir", "type": "STRING", "widget": {"name": "output_dir"}, "link": None},
        {"name": "overwrite", "type": "BOOLEAN", "widget": {"name": "overwrite"}, "link": None},
    ],
    "outputs": [{"name": "captions", "type": "LTX25_CAPTIONS", "links": [6]}],
    "properties": {"Node name for S&R": "O2noorLTX25Int4EncodeCaptions"},
    "widgets_values": ["ltxchar", "0,1", CAPTIONS_DIR, False],
    "title": "LTX 2.5 Int4 Encode Captions",
})
link(6, 3, 4, 1)  # EncodeCaptions -> Dataset.captions

# 4 Dataset
nodes.append({
    "id": 4, "type": "O2noorLTX25Int4Dataset", "pos": [950, 320], "size": [300, 480], "flags": {}, "order": 3, "mode": 0,
    "inputs": [
        {"name": "model", "type": "LTX25_MODEL", "link": 3},
        {"name": "captions", "type": "LTX25_CAPTIONS", "link": 6},
        {"name": "images", "type": "STRING", "widget": {"name": "images"}, "link": None},
        {"name": "voice", "type": "STRING", "widget": {"name": "voice"}, "link": None},
        {"name": "voice_type", "type": "COMBO", "widget": {"name": "voice_type"}, "link": None},
        {"name": "width", "type": "INT", "widget": {"name": "width"}, "link": None},
        {"name": "height", "type": "INT", "widget": {"name": "height"}, "link": None},
        {"name": "frames", "type": "INT", "widget": {"name": "frames"}, "link": None},
        {"name": "clip_fps", "type": "INT", "widget": {"name": "clip_fps"}, "link": None},
        {"name": "trigger_word", "type": "STRING", "widget": {"name": "trigger_word"}, "link": None},
        {"name": "device", "type": "COMBO", "widget": {"name": "device"}, "link": None},
        {"name": "output_dir", "type": "STRING", "widget": {"name": "output_dir"}, "link": None},
        {"name": "vae_tiling", "type": "BOOLEAN", "widget": {"name": "vae_tiling"}, "link": None},
    ],
    "outputs": [{"name": "dataset", "type": "LTX25_DATASET", "links": [7]}],
    "properties": {"Node name for S&R": "O2noorLTX25Int4Dataset"},
    "widgets_values": ["", "", "auto", 512, 512, 25, 24, "ltxchar", "auto", DATASET_DIR, True],
    "title": "LTX 2.5 Int4 Dataset",
})
link(7, 4, 5, 1)  # Dataset -> Train.dataset

# 5 Train
nodes.append({
    "id": 5, "type": "O2noorLTX25Int4Train", "pos": [1400, 200], "size": [380, 380], "flags": {}, "order": 4, "mode": 0,
    "inputs": [
        {"name": "model", "type": "LTX25_MODEL", "link": 4},
        {"name": "dataset", "type": "LTX25_DATASET", "link": 7},
        {"name": "tile_config", "type": "LTX25_TILECONFIG", "link": 5},
        {"name": "run_name", "type": "STRING", "widget": {"name": "run_name"}, "link": None},
        {"name": "steps", "type": "INT", "widget": {"name": "steps"}, "link": None},
        {"name": "lr", "type": "FLOAT", "widget": {"name": "lr"}, "link": None},
        {"name": "rank", "type": "INT", "widget": {"name": "rank"}, "link": None},
        {"name": "alpha", "type": "FLOAT", "widget": {"name": "alpha"}, "link": None},
        {"name": "checkpoint_interval", "type": "INT", "widget": {"name": "checkpoint_interval"}, "link": None},
        {"name": "blocking", "type": "BOOLEAN", "widget": {"name": "blocking"}, "link": None},
    ],
    "outputs": [{"name": "run", "type": "LTX25_RUN", "links": [8, 9]}],
    "properties": {"Node name for S&R": "O2noorLTX25Int4Train"},
    "widgets_values": ["ltx25_train", 2000, 0.0003, 16, 16, 250, True],
    "title": "LTX 2.5 Int4 Train",
})
link(8, 5, 6, 0)  # Train -> Logs
link(9, 5, 7, 0)  # Train -> Validate

# 6 Logs
nodes.append({
    "id": 6, "type": "O2noorLTX25Int4LogsOutputs", "pos": [1840, 40], "size": [460, 380], "flags": {}, "order": 5, "mode": 0,
    "inputs": [{"name": "run", "type": "LTX25_RUN", "link": 8},
               {"name": "tail_lines", "type": "INT", "widget": {"name": "tail_lines"}, "link": None}],
    "outputs": [{"name": "summary", "type": "STRING", "links": []},
                {"name": "milestones", "type": "STRING", "links": []}],
    "properties": {"Node name for S&R": "O2noorLTX25Int4LogsOutputs"},
    "widgets_values": [25],
    "title": "LTX 2.5 Int4 Logs/Output",
})

# 8 Preview Dataset (wired from Dataset, self-displaying)
link(10, 4, 7, 0)  # Dataset -> PreviewDataset
nodes.append({
    "id": 8, "type": "O2noorLTX25Int4PreviewDataset", "pos": [950, 700], "size": [400, 300], "flags": {}, "order": 7, "mode": 0,
    "inputs": [{"name": "dataset", "type": "LTX25_DATASET", "link": 10}],
    "outputs": [{"name": "preview", "type": "IMAGE", "links": []},
                {"name": "info", "type": "STRING", "links": []}],
    "properties": {"Node name for S&R": "O2noorLTX25Int4PreviewDataset"},
    "widgets_values": [],
    "title": "LTX 2.5 Int4 Preview Dataset",
})

# 7 Validate
nodes.append({
    "id": 7, "type": "O2noorLTX25Int4Validate", "pos": [1840, 460], "size": [460, 260], "flags": {}, "order": 6, "mode": 0,
    "inputs": [
        {"name": "run", "type": "LTX25_RUN", "link": 9},
        {"name": "setup", "type": "LTX25_SETUP", "link": None},
        {"name": "prompts", "type": "STRING", "widget": {"name": "prompts"}, "link": None},
        {"name": "inference_steps", "type": "INT", "widget": {"name": "inference_steps"}, "link": None},
        {"name": "width", "type": "INT", "widget": {"name": "width"}, "link": None},
        {"name": "height", "type": "INT", "widget": {"name": "height"}, "link": None},
        {"name": "frames", "type": "INT", "widget": {"name": "frames"}, "link": None},
        {"name": "generate_audio", "type": "BOOLEAN", "widget": {"name": "generate_audio"}, "link": None},
    ],
    "outputs": [{"name": "validation", "type": "STRING", "links": []}],
    "properties": {"Node name for S&R": "O2noorLTX25Int4Validate"},
    "widgets_values": ["A person speaks to the camera.", 20, 512, 512, 25, True],
    "title": "LTX 2.5 Int4 Validate",
})

graph = {"id": str(uuid.uuid4()), "revision": 0,
         "last_node_id": 8, "last_link_id": 10,
         "nodes": nodes, "links": L,
         "groups": [], "config": {}, "extra": {}, "version": 0.4}
for d in (PACK_WF, USER_WF):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "ltx25_int4_train_workflow.json"), "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)
print(f"7-node workflow: {len(nodes)} nodes, {len(L)} links")