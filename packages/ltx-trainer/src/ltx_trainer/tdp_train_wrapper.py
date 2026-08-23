"""Grad-aware Tiled Data Parallel wrapper for LTX-2.5 LoRA TRAINING (custom).

Each training rank holds the FULL model (int4-quanto 22B fits a 12GB 3060)
and processes a spatial tile of the video latent. The blended tile outputs
are summed with an *autograd-aware* all_reduce so the loss sees the full
video; after backward, parameter gradients are all_reduced on CPU (gloo-safe
on Windows, where CUDA collectives hang) so every rank's LoRA stays in sync.

Why this is custom:
  * ltx-core's TiledDataParallelModelWrapper uses raw ``dist.all_reduce`` on
    CUDA tensors (no autograd, and gloo CUDA collectives hang on this box).
  * DDP cannot be used: its automatic CUDA gradient sync also hangs under
    gloo, and it would shard the batch (TDP needs both ranks on the SAME
    sample's tiles). So we do a manual CPU all_reduce of gradients after
    backward.

The wrapper implements the same ``forward(video, audio, perturbations) ->
(video_pred, audio_pred)`` signature the trainer calls, and exposes
``.parameters()``/``.trainable_params`` through to the PEFT/LoRA model.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from ltx_core.model.transformer.modality import Modality
from ltx_core.modality_tiling import VideoModalityTilingHelper
from ltx_core.tiling import TileCountConfig
from ltx_core.tools import VideoLatentTools


class _CPUAllReduce(torch.autograd.Function):
    """Autograd-aware all_reduce that does the collective on CPU (gloo-safe).
    Gradient of a SUM is the same on every rank, so backward just passes the
    incoming gradient through unchanged.
    """

    @staticmethod
    def forward(ctx, x):  # type: ignore[override]
        x_cpu = x.detach().cpu()
        dist.all_reduce(x_cpu)
        return x_cpu.to(x.device)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        return grad_output


def tdp_cpu_all_reduce(x: torch.Tensor) -> torch.Tensor:
    """Autograd-aware all_reduce (CPU collective) for the blend output."""
    return _CPUAllReduce.apply(x)


def tdp_sync_grads(model: nn.Module) -> None:
    """Sum every trainable parameter's gradient across ranks (CPU collective)."""
    if not dist.is_initialized() or dist.get_world_size() < 2:
        return
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            g = p.grad.detach().cpu()
            dist.all_reduce(g)
            p.grad.copy_(g.to(p.device))


class GradTiledDataParallelModelWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        video_tools: VideoLatentTools,
        tiling: TileCountConfig,
        train_group: list[int] | None = None,
        normalize_positions: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self._helper = VideoModalityTilingHelper(tiling, video_tools)
        all_tiles = self._helper.tiles
        members = sorted(train_group) if train_group else None
        if members is None or dist.get_rank() in members:
            r = dist.get_rank()
            self._tiles = [t for i, t in enumerate(all_tiles) if i % len(members) == members.index(r)]
        else:
            self._tiles = []
        self._normalize_positions = normalize_positions

    @property
    def num_blocks(self) -> int:
        return self.model.num_blocks

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: object | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if video is None or not self._tiles:
            return self.model(video, audio, perturbations)

        denoised_video: torch.Tensor | None = None
        denoised_audio: torch.Tensor | None = None
        for tile in self._tiles:
            tiled_video, ctx = self._helper.tile_modality(video, tile, normalize_positions=self._normalize_positions)
            tile_out, audio_out = self.model(tiled_video, audio, perturbations)
            blended = self._helper.blend(tile_out, tile, ctx)
            denoised_video = blended if denoised_video is None else denoised_video + blended
            if audio_out is not None:
                denoised_audio = audio_out if denoised_audio is None else denoised_audio + audio_out

        assert denoised_video is not None
        # Autograd-aware all_reduce so the loss sees the full video and gradients
        # flow back through the blend into this rank's LoRA params.
        denoised_video = tdp_cpu_all_reduce(denoised_video.contiguous())

        if denoised_audio is not None:
            total_tiles = len(self._helper.tiles)
            denoised_audio = tdp_cpu_all_reduce(denoised_audio.contiguous()) / total_tiles

        return denoised_video, denoised_audio
