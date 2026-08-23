"""Reusable spatial-tiling feature for the sharded int4 engine.

Composes with N-GPU weight sharing: each rank holds a block shard, and spatial
tiling splits the latent so activation memory scales down by tile count. Tiling
happens at the MODALITY level (VideoModalityTilingHelper) so each tile carries
correctly-rebased positions -> valid per-tile RoPE.

Reusable by the future ComfyUI node: this module is GPU/world-agnostic and takes
plain (modality, TilingConfig, video_tools).
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass

import torch  # noqa: E402

from ltx_core.modality_tiling import VideoModalityTilingHelper  # noqa: E402
from ltx_core.tiling import DimensionTilingConfig, TileCountConfig  # noqa: E402

MAX_GRID = 6
MAX_OVERLAP = 16


@dataclass(frozen=True)
class TilingConfig:
    """Grid ('default'|'HxW') and per-axis overlap (0..16). 'default' == no tiling."""

    grid: str = "default"
    overlap: int = 0

    def is_tiled(self) -> bool:
        return self.grid != "default"

    def grid_dims(self) -> tuple[int, int]:
        """Return (rows, cols) or (1,1) for default."""
        if self.grid == "default":
            return (1, 1)
        m = re.fullmatch(r"(\d+)x(\d+)", self.grid.strip().lower())
        if not m:
            raise ValueError(f"invalid grid {self.grid!r}; use HxW e.g. 2x2 or 'default'")
        h, w = int(m.group(1)), int(m.group(2))
        if h < 1 or w < 1 or h > MAX_GRID or w > MAX_GRID:
            raise ValueError(f"grid dims must be 1..{MAX_GRID}, got {h}x{w}")
        return (h, w)


def resolve_tiling(grid: str, overlap: int, latent_h: int, latent_w: int) -> TilingConfig:
    """Build a TilingConfig with overlap clamped to < tile size per axis (keeps tiling valid)."""
    ov = max(0, min(MAX_OVERLAP, int(overlap)))
    cfg = TilingConfig(grid=grid, overlap=ov)
    if not cfg.is_tiled():
        return cfg
    rows, cols = cfg.grid_dims()
    if rows > latent_h or cols > latent_w:
        raise ValueError(f"tile grid {rows}x{cols} exceeds latent {latent_h}x{latent_w}")
    th = (latent_h + rows - 1) // rows
    tw = (latent_w + cols - 1) // cols
    # clamp overlap to < tile size so trapezoidal masks stay valid
    ov = min(ov, th - 1, tw - 1)
    return TilingConfig(grid=grid, overlap=max(0, ov))


def build_tiles(modality, cfg: TilingConfig, video_tools):
    """Return (tiles, contexts, helper).
    tiles[i] is the tiled Modality for helper.tiles[i], contexts[i] its TilingContext.
    Untiled ('default') returns a single identity tile list."""
    tiling = TileCountConfig(
        height=DimensionTilingConfig(1, 0),
        width=DimensionTilingConfig(1, 0),
    )
    if cfg.is_tiled():
        rows, cols = cfg.grid_dims()
        tiling = TileCountConfig(
            height=DimensionTilingConfig(rows, cfg.overlap),
            width=DimensionTilingConfig(cols, cfg.overlap),
        )
    helper = VideoModalityTilingHelper(tiling, video_tools)
    pairs = [helper.tile_modality(modality, t, normalize_positions=False) for t in helper.tiles]
    tiles = [p[0] for p in pairs]
    ctxs = [p[1] for p in pairs]
    return tiles, ctxs, helper


def blend_tiles(helper, tile_outputs, ctxs):
    """Blend per-tile denoised outputs (B, tile_tokens, D) into full (B, total_tokens, D).
    tile_outputs must align with helper.tiles; ctxs from build_tiles."""
    out = None
    for i, (tile, ctx) in enumerate(zip(helper.tiles, ctxs)):
        tile_out = tile_outputs[i]
        out = helper.blend(tile_out, tile, ctx, output=out)
    return out
