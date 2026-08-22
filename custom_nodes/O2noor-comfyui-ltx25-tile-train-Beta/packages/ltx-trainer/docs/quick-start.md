# Quick Start Guide

Get up and running with LTX family training in just a few steps.

## 📋 Prerequisites

Before you begin, ensure you have:

1. **LTX Model Checkpoint** - A local `.safetensors` file containing the model weights.
   The trainer supports LTX-2, LTX-2.3, and **LTX 2.5** through the same configuration API;
   version-specific components are detected from checkpoint metadata.
2. **Matching Gemma Text Encoder** - A local directory containing the Gemma model.
   LTX 2.5 requires the LTX-specific fine-tuned Gemma 4 root (for example, `gemma4-12b-ltx-v1`), not Google's
   vanilla Gemma 4 model. Older LTX-2 and LTX-2.3 checkpoints use the Gemma version they declare; do not substitute
   Gemma 3 for an LTX 2.5 checkpoint.
3. **Linux with CUDA** - The trainer requires `triton` which is Linux-only; CUDA 13+ is recommended
4. **GPU with sufficient VRAM** - 80GB recommended for the standard config. For GPUs with 32GB VRAM (e.g., RTX 5090),
   use the [low VRAM config](../configs/t2v_lora_low_vram.yaml) which enables INT8 quantization and other
   memory optimizations

## ⚡ Installation

First, install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already.
Then clone the repository and install the dependencies:

```bash
git clone https://github.com/Lightricks/LTX-2
```

The `ltx-trainer` package is part of the `LTX-2` monorepo. Install the dependencies from the repository root,
then navigate to the trainer package:

```bash
# From the repository root
uv sync
cd packages/ltx-trainer
```

> [!NOTE]
> The trainer depends on [`ltx-core`](../../ltx-core/) and [`ltx-pipelines`](../../ltx-pipelines/)
> packages which are automatically installed from the monorepo.

## 🏋 Training Workflow

If you are using an agent-enabled environment with repository skills, you can ask for the
[`train-model`](../../../.claude/skills/train-model/SKILL.md) skill to run this workflow with you.
It creates a run workspace, confirms the training mode, prepares data, preprocesses latents,
launches training, and monitors the run while stopping for approval before expensive steps.

### 1. Choose a Training Mode

Start with [`t2v_lora.yaml`](../configs/t2v_lora.yaml) for a first run with videos and captions. For modes such as
IC-LoRA, inpainting, or outpainting, check [Training Modes](training-modes.md) first because your metadata needs extra
columns such as `reference_video`, `video_mask`, or `audio_mask` before preprocessing.

### 2. Prepare Your Dataset

Organize your videos and captions, then preprocess them:

```bash
# Split long videos into scenes (optional)
uv run python scripts/split_scenes.py input.mp4 scenes_output_dir/ --filter-shorter-than 5s

# Generate captions for videos (optional)
uv run python scripts/caption_videos.py scenes_output_dir/ --output dataset.json

# Preprocess the dataset (compute latents and embeddings)
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/ltx-2.x-checkpoint.safetensors \
    --text-encoder-path /path/to/gemma-root
```

By default, preprocessing writes to `.precomputed/`. Use that directory as `data.preprocessed_data_root`
in your training config.

> [!IMPORTANT]
> Precomputed text features are specific to the checkpoint and Gemma pair. When switching from LTX-2.3 to
> LTX 2.5, preprocess into a fresh output directory or add `--overwrite`; existing `.pt` files are skipped by
> default. Use the checkpoint metadata supplied with the model; no version flag is required.

See [Dataset Preparation](dataset-preparation.md) for detailed instructions.

### 3. Configure Training

Create or modify a configuration YAML file. Start with one of the example configs:

- [`configs/t2v_lora.yaml`](../configs/t2v_lora.yaml) - Text-to-video LoRA
- [`configs/t2v_lora_low_vram.yaml`](../configs/t2v_lora_low_vram.yaml) - Same as above, tuned for ~32GB VRAM (INT8 quantization and memory optimizations)
- [`configs/v2v_ic_lora.yaml`](../configs/v2v_ic_lora.yaml) - IC-LoRA video-to-video

Key settings to update:

A unified (single-file) checkpoint needs two paths:

```yaml
model:
  model_path: "/path/to/ltx-2.x-checkpoint.safetensors"
  text_encoder_path: "/path/to/gemma-root"

data:
  preprocessed_data_root: "/path/to/preprocessed/data"

output_dir: "outputs/my_training_run"
```

A split pack ships one file per component, so name each of them instead:

```yaml
model:
  model_path: "/path/to/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
  text_encoder_path: "/path/to/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
  video_vae_path: "/path/to/vae/ltx-2.5-video-vae-bf16.safetensors"
  audio_vae_path: "/path/to/vae/ltx-2.5-audio-vae-bf16.safetensors"
```

See [Configuration Reference](configuration-reference.md) for all available options.

### 4. Start Training

```bash
uv run python scripts/train.py configs/t2v_lora.yaml
```

For multi-GPU training:

```bash
uv run accelerate launch scripts/train.py configs/t2v_lora.yaml
```

See [Training Guide](training-guide.md) for distributed training and advanced options.

## 🎯 Training Modes

> [!TIP]
> **First time?** Start with [`t2v_lora.yaml`](../configs/t2v_lora.yaml) — it's the simplest mode
> and only requires videos with captions. You can explore other modes once you've confirmed your
> setup works.

The trainer supports several training modes:

| Mode                  | Description                                | Example Config                                                    |
|-----------------------|--------------------------------------------|-------------------------------------------------------------------|
| **Text-to-Video**     | Generate video+audio from text prompts     | [`t2v_lora.yaml`](../configs/t2v_lora.yaml)                      |
| **Image-to-Video**    | Animate from a starting image              | [`i2v_lora.yaml`](../configs/i2v_lora.yaml)                      |
| **Video Extension**   | Extend videos temporally (forward/backward)| [`video_extend_lora.yaml`](../configs/video_extend_lora.yaml), [`video_suffix_lora.yaml`](../configs/video_suffix_lora.yaml) |
| **IC-LoRA (V2V)**     | Video-to-video transformations             | [`v2v_ic_lora.yaml`](../configs/v2v_ic_lora.yaml)                |
| **Audio-to-Video**    | Generate video conditioned on audio        | [`a2v_lora.yaml`](../configs/a2v_lora.yaml)                      |
| **Video-to-Audio**    | Generate audio/foley from video            | [`v2a_lora.yaml`](../configs/v2a_lora.yaml)                      |
| **Video Inpainting**  | Fill in masked regions of video            | [`video_inpainting_lora.yaml`](../configs/video_inpainting_lora.yaml) |
| **Video Outpainting** | Extend video spatially                     | [`video_outpainting_lora.yaml`](../configs/video_outpainting_lora.yaml) |
| **Text-to-Audio**     | Generate audio from text prompts           | [`t2a_lora.yaml`](../configs/t2a_lora.yaml)                      |
| **Audio Extension**   | Extend audio temporally                    | [`audio_extend_lora.yaml`](../configs/audio_extend_lora.yaml), [`audio_suffix_lora.yaml`](../configs/audio_suffix_lora.yaml) |
| **Audio Inpainting**  | Fill in masked regions of audio            | [`audio_inpainting_lora.yaml`](../configs/audio_inpainting_lora.yaml) |
| **IC-LoRA (A2A)**     | Audio-to-audio transformations             | [`a2a_ic_lora.yaml`](../configs/a2a_ic_lora.yaml)                |
| **AV2AV IC-LoRA**     | Audio+video IC-LoRA transformations        | [`av2av_ic_lora.yaml`](../configs/av2av_ic_lora.yaml)            |
| **Full Fine-tuning**  | Full model training (any mode above)       | Set `model.training_mode: "full"`                                 |

See [Training Modes](training-modes.md) for detailed explanations of each mode.

## Next Steps

Once you've completed your first training run, you can:

- **Use your trained LoRA for inference** - The [`ltx-pipelines`](../../ltx-pipelines/) package provides
  production-ready inference
  pipelines for various use cases (T2V, I2V, IC-LoRA, etc.). See the package documentation for details.
- Learn more about [Dataset Preparation](dataset-preparation.md) for advanced preprocessing
- Explore different [Training Modes](training-modes.md)
- Dive deeper into [Training Configuration](configuration-reference.md)
- Understand the model architecture in [LTX-Core Documentation](../../ltx-core/README.md)

## Need Help?

If you run into issues at any step, see the [Troubleshooting Guide](troubleshooting.md) for solutions to common
problems.

Join our [Discord community](https://discord.gg/ltxplatform) for real-time help and discussion!
