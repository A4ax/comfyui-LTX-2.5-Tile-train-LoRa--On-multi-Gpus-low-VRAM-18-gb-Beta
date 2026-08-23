"""O2noorLTX25Int4TileConfig — activation tiling for training (takes the MODEL).

Configures horizontal/vertical tiling + overlap consumed by the Train node.
Per-GPU block allocation is NOT set here — it lives on the Load Model node
(transformer_blocks_gpu*) and overrides anything set here.
"""


class O2noorLTX25Int4TileConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LTX25_MODEL", {"tooltip": "The int4 model from O2noorLTX25Int4LoadModel."}),
                "horizontal_tiles": ("INT", {
                    "default": 1, "min": 1, "max": 6, "step": 1,
                    "tooltip": "Number of tiles across the WIDTH (1 = no horizontal tiling).",
                }),
                "vertical_tiles": ("INT", {
                    "default": 1, "min": 1, "max": 6, "step": 1,
                    "tooltip": "Number of tiles down the HEIGHT (1 = no vertical tiling).",
                }),
                "overlap": ("INT", {
                    "default": 0, "min": 0, "max": 16, "step": 1,
                    "tooltip": "Per-axis tile overlap in latent grid units (0-16). Blend zones between tiles.",
                }),
            }
        }

    RETURN_TYPES = ("LTX25_TILECONFIG",)
    RETURN_NAMES = ("tile_config",)
    FUNCTION = "build"
    CATEGORY = "ltx25-int4-train"
    TITLE = "O2noor LTX 2.5 Int4 Tile Config"

    def build(self, model, horizontal_tiles, vertical_tiles, overlap):
        cfg = {
            "horizontal_tiles": max(1, min(6, int(horizontal_tiles))),
            "vertical_tiles": max(1, min(6, int(vertical_tiles))),
            "overlap": max(0, min(16, int(overlap))),
        }
        print(f"[O2noorLTX25Int4TileConfig] {cfg['horizontal_tiles']}x{cfg['vertical_tiles']} ov{cfg['overlap']}", flush=True)
        return (cfg,)
