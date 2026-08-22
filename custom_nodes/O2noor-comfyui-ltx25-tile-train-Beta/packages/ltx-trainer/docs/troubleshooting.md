# Troubleshooting Guide

This guide covers common issues and solutions when training with the LTX-2 trainer.

## 🔧 VRAM and Memory Issues

Memory management is crucial for successful training with LTX-2.

> [!TIP]
> For GPUs with 32GB VRAM, use the pre-configured low VRAM config:
> [`configs/t2v_lora_low_vram.yaml`](../configs/t2v_lora_low_vram.yaml)
> which combines 8-bit optimizer, INT8 quantization, and reduced LoRA rank.

### Memory Optimization Techniques

#### 1. Enable Gradient Checkpointing

Gradient checkpointing trades training speed for memory savings. **Highly recommended** for most training runs:

```yaml
optimization:
  enable_gradient_checkpointing: true
```

#### 2. Enable 8-bit Text Encoder

Quantize the Gemma text encoder to 8-bit, roughly halving the VRAM it holds while caption embeddings are computed:

```yaml
acceleration:
  load_text_encoder_in_8bit: true
```

This works with both checkpoint layouts. It requires `bitsandbytes` and a CUDA device, and the full-precision
weights pass through host RAM on the way to being quantized — budget around 26 GB of RAM for LTX 2.5.


#### 3. Reduce Batch Size

Lower the batch size if you encounter out-of-memory errors:

```yaml
optimization:
  batch_size: 1  # Start with 1 and increase gradually
```

Use gradient accumulation to maintain a larger effective batch size:

```yaml
optimization:
  batch_size: 1
  gradient_accumulation_steps: 4  # Effective batch size = 4
```

#### 4. Use Lower Resolution

Reduce spatial or temporal dimensions to save memory:

```bash
# Smaller spatial resolution
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets "512x512x49" \
    --model-path /path/to/model.safetensors \
    --text-encoder-path /path/to/gemma

# Fewer frames
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x25" \
    --model-path /path/to/model.safetensors \
    --text-encoder-path /path/to/gemma
```

#### 5. Enable Model Quantization

Use quantization to reduce memory usage:

```yaml
acceleration:
  quantization: "int8-quanto"  # Options: int8-quanto, int4-quanto, fp8-quanto
```

#### 6. Use 8-bit Optimizer

The 8-bit AdamW optimizer uses less memory:

```yaml
optimization:
  optimizer_type: "adamw8bit"
```

#### 7. Offload Optimizer State During Validation

If you OOM specifically during validation video sampling — typically in
full fine-tunes or high-rank LoRA runs where AdamW state and the VAE decoder
can't coexist on the GPU — offload optimizer state to CPU during sampling:

```yaml
acceleration:
  offload_optimizer_during_validation: true
```

The offload + reload happens once per validation interval, not per step.
No effect for FSDP (sharded state).

---

## ⚠️ Common Usage Issues

### Issue: "No module named 'ltx_trainer'" Error

**Solution:**
Ensure you've installed the dependencies and are using `uv run` to execute scripts:

```bash
# From the repository root
uv sync
cd packages/ltx-trainer
uv run python scripts/train.py configs/t2v_lora.yaml
```

> [!TIP]
> Always use `uv run` to execute Python scripts. This automatically uses the correct virtual environment
> without requiring manual activation.

### Issue: Wrong `text_encoder_path` for the checkpoint layout

**Solution:**
`text_encoder_path` follows the layout of the model you are training on. With a unified checkpoint it is a
directory holding the matching Gemma model; with a split pack it is the packed text-encoder `.safetensors`,
which embeds the Gemma weights, the HF sidecars and the text projections in one file. LTX 2.5 requires the
LTX-specific fine-tuned Gemma 4, not Google's vanilla Gemma 4 model; do not reuse the Gemma 3 root from an
LTX-2.3 run.

```yaml
model:
  model_path: "/path/to/ltx-2.x-checkpoint.safetensors"  # Unified checkpoint
  text_encoder_path: "/path/to/matching-gemma-root/"      # Gemma directory
```

### Issue: "is the transformer of a split checkpoint pack and carries no video VAE" Error

**Solution:**
`model_path` points at a split pack's transformer, which holds no VAE weights. Name the components explicitly:

```yaml
model:
  model_path: "/path/to/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
  text_encoder_path: "/path/to/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
  video_vae_path: "/path/to/vae/ltx-2.5-video-vae-bf16.safetensors"
  audio_vae_path: "/path/to/vae/ltx-2.5-audio-vae-bf16.safetensors"
```

The preprocessing scripts take the same components as `--video-vae-path` / `--audio-vae-path`.

### Issue: "Gemma version mismatch" Error

**Solution:**

The checkpoint and Gemma root are incompatible. Use the LTX-specific fine-tuned Gemma 4 root named by the checkpoint's
`gemma_source_checkpoint` metadata and do not mix LTX-2.3 caption features or Gemma weights with LTX 2.5.
The checkpoint metadata supplied with the model is authoritative.

### Issue: Training fails after switching from LTX-2.3 to LTX 2.5

**Solution:**

Recompute the preprocessed data. Caption feature files in `conditions/` are generated by the selected Gemma model
and are not interchangeable across versions:

```bash
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/ltx-2.x-checkpoint.safetensors \
    --text-encoder-path /path/to/gemma4-12b-ltx-v1 \
    --overwrite
```

Use a fresh output directory when possible. `process_dataset.py`, `process_captions.py`, and `process_videos.py`
skip existing `.pt` files unless `--overwrite` is passed.

### Issue: "Model path does not exist" Error

**Solution:**
LTX-2 requires local model paths. URLs are not supported:

```yaml
# ✅ Correct - local path
model:
  model_path: "/path/to/ltx-2-model.safetensors"

# ❌ Wrong - URL not supported
model:
  model_path: "https://huggingface.co/..."
```

### Issue: "frames must satisfy (frames - 1) % T == 0" Error

**Solution:**
The number of frames must be VAE-aligned: `frames % T == 1`, where `T` is the VAE temporal
compression factor read from the checkpoint. For the **default** VAE, `T = 8`:

- ✅ Valid: 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 121
- ❌ Invalid: 24, 32, 48, 64, 100

A less-compressed VAE lowers `T` (e.g. a 16x16x4 VAE uses `T = 4`, allowing 1, 5, 9, 13, ...).
The spatial dimensions must likewise be divisible by the VAE spatial factor (32 by default).

### Issue: Slow Training Speed

**Optimizations:**

1. **Disable gradient checkpointing** (if you have enough VRAM):

   ```yaml
   optimization:
     enable_gradient_checkpointing: false
   ```


2. **Use torch.compile** via Accelerate:

   ```bash
   uv run accelerate launch --config_file configs/accelerate/ddp_compile.yaml \
     scripts/train.py configs/t2v_lora.yaml
   ```

### Issue: Poor Quality Validation Outputs

**Solutions:**

First confirm that the validation recipe matches the checkpoint. The trainer's validation runner is one-stage:
for an LTX 2.5 distilled checkpoint, start with 8 steps, CFG scales of `1.0`, and STG scales of `0.0`. PT/SFT
checkpoints need their own one-stage sampling recipe. The production two-stage distilled workflow is provided by
[`ltx-pipelines`](../../ltx-pipelines/README.md), not this training-time validator.

1. **Use conditioned validation:** For more reliable validation, use image-to-video (first-frame conditioning) rather than pure text-to-video:

   ```yaml
   validation:
     samples:
       - prompt: "a professional portrait video of a person"
         conditions:
           - type: first_frame
             image_or_video: "/path/to/first_frame.png"
   ```

2. **Increase inference steps:**

   ```yaml
   validation:
     inference_steps: 30
   ```

3. **Adjust guidance settings:**

   ```yaml
   validation:
     video_cfg_scale: 3.0
     audio_cfg_scale: 7.0
     video_stg_scale: 1.0
     audio_stg_scale: 1.0
     stg_blocks: [28]
     guidance_rescale: 0.7
     video_modality_guidance_scale: 3.0
     audio_modality_guidance_scale: 3.0
   ```

4. **Check caption quality:**
   Review and manually edit captions for accuracy if using auto-generated captions.
   LTX models prefer long, detailed captions that describe both visual content and audio (e.g., ambient sounds, speech,
   music).

5. **Check target modules:**
   Ensure your `target_modules` configuration matches your training goals. For audio-video training,
   use patterns that match both branches (e.g., `"to_k"` instead of `"attn1.to_k"`).
   See [Understanding Target Modules](configuration-reference.md#understanding-target-modules) for details.

6. **Adjust LoRA rank:**
   Try higher values for more capacity:

   ```yaml
   lora:
     rank: 64  # Or 128 for more capacity
   ```

7. **Increase training steps:**

   ```yaml
   optimization:
     steps: 3000
   ```

---

## 🔍 Debugging Tools

### Monitor GPU Memory Usage

Track memory usage during training:

```bash
# Watch GPU memory in real-time
watch -n 1 nvidia-smi

# Log memory usage to file
nvidia-smi --query-gpu=memory.used,memory.total --format=csv --loop=5 > memory_log.csv
```

### Verify Preprocessed Data

Decode latents to visualize the preprocessed videos:

```bash
uv run python scripts/decode_latents.py dataset/.precomputed/latents debug_output \
    --model-path /path/to/model.safetensors
```

To also decode audio latents, add the `--with-audio` flag:

```bash
uv run python scripts/decode_latents.py dataset/.precomputed/latents debug_output \
    --model-path /path/to/model.safetensors \
    --with-audio
```

Compare decoded videos and audio with originals to ensure quality.

---

## 💡 Best Practices

### Before Training

- [ ] Test preprocessing with a small subset first
- [ ] Verify all video files are accessible
- [ ] Check available GPU memory
- [ ] Review configuration against hardware capabilities
- [ ] Ensure model and text encoder paths are correct

### During Training

- [ ] Monitor GPU memory usage
- [ ] Check loss convergence regularly
- [ ] Review validation samples periodically
- [ ] Save checkpoints frequently

### After Training

- [ ] Test trained model with diverse prompts
- [ ] Document training parameters and results
- [ ] Archive training data and configs

## 🆘 Getting Help

If you're still experiencing issues:

1. **Check logs:** Review console output for error details
2. **Search issues:** Look through GitHub issues for similar problems
3. **Provide details:** When reporting issues, include:
    - Hardware specifications (GPU model, VRAM)
    - Configuration file used
    - Complete error message
    - Steps to reproduce the issue

---

## 🤝 Join the Community

Have questions, want to share your results, or need real-time help?
Join our [community Discord server](https://discord.gg/ltxplatform)
to connect with other users and the development team!

- Get troubleshooting help
- Share your training results and workflows
- Stay up to date with announcements and updates

We look forward to seeing you there!
