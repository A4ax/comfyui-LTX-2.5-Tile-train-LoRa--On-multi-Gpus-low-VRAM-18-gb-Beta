# Utility Scripts Reference

This guide covers the various utility scripts available for preprocessing, conversion, and debugging tasks.

All model-processing scripts use the same automatic checkpoint detection as the trainer and support LTX-2, LTX-2.3, and LTX 2.5.
The checkpoint and Gemma paths must refer to a compatible pair.

## 🎬 Dataset Processing Scripts

### Video Scene Splitting

The `scripts/split_scenes.py` script automatically splits long videos into shorter, coherent scenes.

```bash
# Basic scene splitting
uv run python scripts/split_scenes.py input.mp4 output_dir/ --filter-shorter-than 5s
```

**Key features:**

- **Automatic scene detection**: Uses PySceneDetect for intelligent splitting
- **Multiple algorithms**: Content-based, adaptive, threshold, and histogram detection
- **Filtering options**: Remove scenes shorter than specified duration
- **Customizable parameters**: Thresholds, window sizes, and detection modes

**Common options:**

```bash
# See all available options
uv run python scripts/split_scenes.py --help

# Use adaptive detection with custom threshold
uv run python scripts/split_scenes.py video.mp4 scenes/ --detector adaptive --threshold 30.0

# Limit to maximum number of scenes
uv run python scripts/split_scenes.py video.mp4 scenes/ --max-scenes 50
```

### Automatic Video Captioning

The `scripts/caption_videos.py` script generates a single, detailed combined audio-visual
caption per video as a continuous paragraph of prose. Two backends are available:

- **`qwen_omni` (default)** — Qwen3-Omni-30B-A3B-Thinking served via a local
  [vLLM](https://docs.vllm.ai/) HTTP server (~1-3 s/video on H100). Highest quality, runs
  fully offline once the model is downloaded.
- **`gemini_flash`** — Google Gemini (cloud, `gemini-3.5-flash`). No GPU required. Auth is
  automatic: set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) for the Developer API, or just have
  Google Cloud credentials available (`gcloud auth` / an attached service account) and it
  uses Vertex AI with no extra setup.

**Step 1 — launch the captioner server** (`qwen_omni` only, one-time).

`scripts/serve_captioner.py` runs vLLM in an isolated environment via `uvx`, so vLLM's heavy
CUDA dependencies never touch the trainer's venv. It defaults to dynamic FP8 quantization
(~31 GiB weights, fits on 40 GB GPUs, same speed as BF16 on H100):

```bash
# Terminal 1 - stays running
uv run python packages/ltx-trainer/scripts/serve_captioner.py

# Useful variants:
#   --print-cmd           show the vLLM command without running it
#   --quantization bf16   use BF16 instead (needs ~66 GiB free VRAM)
#   --hf-home /mnt/disk   override where the ~65 GB model is downloaded
```

**Step 2 — caption your videos.**

```bash
# Terminal 2 - default backend talks to the server above
uv run python packages/ltx-trainer/scripts/caption_videos.py videos_dir/ --output dataset.json

# Remote server:           --vllm-url http://other-host:8001/v1
# Gemini (gemini-3.5-flash): --captioner-type gemini_flash   (uses GEMINI_API_KEY, else gcloud/Vertex)
# Gemini, parallel calls:  --captioner-type gemini_flash --num-workers 5
# Re-caption everything:   --override
```

Captioning is incremental (already-captioned files are skipped, progress saves every 5 videos)
and writes JSON, JSONL, CSV, or TXT based on the output extension.

Qwen3-Omni-Thinking can optionally emit a `<think>...</think>` chain-of-thought before the
caption (`--enable-thinking`). It is off by default, which is recommended for bulk captioning
(thinking is slower as it generates the reasoning trace first).

For Gemini, keep `--num-workers` at 3-5 (higher values may hit API rate limits).

### Dataset Preprocessing

The `scripts/process_dataset.py` script processes videos and caches latents for training.

> [!IMPORTANT]
> The examples below use a **unified checkpoint** — one `.safetensors` holding every component, plus a Gemma folder.
> On a **split pack** (LTX 2.5, one file per component) the transformer holds no VAE weights, so every command that
> encodes latents also needs `--video-vae-path` and `--audio-vae-path`:
>
> ```bash
> uv run python scripts/process_dataset.py dataset.json \
>     --resolution-buckets "960x544x49" \
>     --model-path /path/to/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
>     --text-encoder-path /path/to/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
>     --video-vae-path /path/to/vae/ltx-2.5-video-vae-bf16.safetensors \
>     --audio-vae-path /path/to/vae/ltx-2.5-audio-vae-bf16.safetensors
> ```
>
> The same two flags apply to `process_videos.py` and `decode_latents.py`. `process_captions.py` does not need them:
> it only runs the text encoder.

```bash
# Basic preprocessing
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/ltx-2.x-checkpoint.safetensors \
    --text-encoder-path /path/to/gemma-root

# With video decoding for verification
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/ltx-2.x-checkpoint.safetensors \
    --text-encoder-path /path/to/gemma-root \
    --decode
```

Multiple resolution buckets can be specified, separated by `;`:

```bash
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49;512x512x81" \
    --model-path /path/to/ltx-2.x-checkpoint.safetensors \
    --text-encoder-path /path/to/gemma-root
```

> [!NOTE]
> When training with multiple resolution buckets, set `optimization.batch_size: 1`.
> When switching between LTX-2.3 and LTX 2.5, use a fresh output directory or `--overwrite`; existing text-feature
> `.pt` files are skipped by default and are not interchangeable between Gemma versions.

**Multi-GPU preprocessing.** Launch with `accelerate launch` to shard the dataset across processes. Reruns resume
by default (existing `.pt` outputs are skipped); writes are atomic so interrupted runs are safe. Pass `--overwrite`
when rerunning with changed parameters (different model, resolution buckets, text encoder, `--lora-trigger`, etc.)
so stale outputs are replaced. Use the same `accelerate launch` pattern (and `--overwrite` when needed) with
`process_videos.py` or `process_captions.py` when you run those scripts standalone.

```bash
# Multi-GPU preprocessing
uv run accelerate launch --num_processes 4 scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/ltx-2.x-checkpoint.safetensors \
    --text-encoder-path /path/to/gemma-root

# Force re-encoding of all items (e.g. after switching model or resolution)
uv run accelerate launch --num_processes 4 scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/ltx-2.x-checkpoint.safetensors \
    --text-encoder-path /path/to/gemma-root \
    --overwrite
```

For detailed usage, see the [Dataset Preparation Guide](dataset-preparation.md).

### Reference Video Generation

The `scripts/compute_reference.py` script provides a template for creating reference videos needed for IC-LoRA training.
The default implementation generates Canny edge reference videos.

```bash
# Generate Canny edge reference videos
uv run python scripts/compute_reference.py videos_dir/ --output dataset.json
```

**Key features:**

- **Canny edge detection**: Creates edge-based reference videos
- **In-place editing**: Updates existing dataset JSON files
- **Customizable**: Modify the `compute_reference()` function for different conditions (depth, pose, etc.)

> [!TIP]
> You can edit this script to generate other types of reference videos for IC-LoRA training,
> such as depth maps, segmentation masks, or any custom video transformation.

> [!NOTE]
> `compute_reference.py` writes generated references to the `reference_video` column, which
> `process_dataset.py` detects automatically.

## 🔍 Debugging and Verification Scripts

### Latents Decoding

The `scripts/decode_latents.py` script decodes precomputed video latents back into video files for visual inspection.

```bash
# Basic usage
uv run python scripts/decode_latents.py /path/to/latents/dir /path/to/output \
    --model-path /path/to/ltx-2.x-checkpoint.safetensors

# Decode both video and audio latents
uv run python scripts/decode_latents.py /path/to/latents/dir /path/to/output \
    --model-path /path/to/ltx-2.x-checkpoint.safetensors \
    --with-audio
```

**The script will:**

1. **Load the VAE model** from the specified path
2. **Process all `.pt` latent files** in the input directory
3. **Decode each latent** back into a video using the VAE
4. **Save resulting videos** as MP4 files in the output directory

**When to use:**

- **Verify preprocessing quality**: Check that your videos were encoded correctly
- **Debug training data**: Visualize what the model actually sees during training
- **Quality assessment**: Ensure latent encoding preserves important visual details

### Inference with Trained Models

For inference with trained LoRAs, use the [`ltx-pipelines`](../../ltx-pipelines/) package which provides
production-ready pipelines:

- **Text/Image-to-Video**: `TI2VidOneStagePipeline`, `TI2VidTwoStagesPipeline`
- **Distilled (fast) inference**: `DistilledPipeline`
- **IC-LoRA video-to-video**: `ICLoraPipeline`
- **Keyframe interpolation**: `KeyframeInterpolationPipeline`

All pipelines support loading custom LoRAs trained with this trainer.

## 🚀 Training Scripts

### Basic and Distributed Training

Use `scripts/train.py` for both single GPU and multi-GPU runs:

```bash
# Single-GPU training
uv run python scripts/train.py configs/t2v_lora.yaml

# Multi-GPU (uses your accelerate config)
uv run accelerate launch scripts/train.py configs/t2v_lora.yaml

# Override number of processes
uv run accelerate launch --num_processes 4 scripts/train.py configs/t2v_lora.yaml
```

For detailed usage, see the [Training Guide](training-guide.md).

## 💡 Tips for Using Utility Scripts

- **Start with `--help`**: Always check available options for each script
- **Test on small datasets**: Verify workflows with a few files before processing large datasets
- **Use decode verification**: Always decode a few samples to verify preprocessing quality
- **Monitor VRAM usage**: Reach for quantization or lower-memory settings (e.g. FP8 for the captioner server) when running into memory issues
- **Keep backups**: Make copies of important dataset files before running conversion scripts
