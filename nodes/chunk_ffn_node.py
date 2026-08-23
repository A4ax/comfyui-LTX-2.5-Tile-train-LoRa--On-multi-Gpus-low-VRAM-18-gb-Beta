"""O2noorLTX25ChunkFeedForward — training-only low-VRAM feed-forward chunking.

Modified from kjNodes' "LTXV Chunk FeedForward" node, rewritten backward-safe so
it works in the TRAINING engine (the original's in-place write breaks autograd).

Chunks each transformer block's feed-forward over the token/sequence dimension,
so the FFN never materializes the full activation at once -> lower peak
activation VRAM during training (higher resolution / longer clips). `chunks=1`
is a no-op. This is a MODEL->MODEL node: it takes the int4 model and returns it
(with the chunking config stamped on it). Feed the output into the Train node's
`model` input.
"""


class O2noorLTX25ChunkFeedForward:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LTX25_MODEL", {"tooltip": "The int4 model from O2noorLTX25Int4LoadModel."}),
                "chunks": ("INT", {
                    "default": 2, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Number of chunks to split the feed-forward activations into "
                               "(1 = no chunking / disabled).",
                }),
                "dim_threshold": ("INT", {
                    "default": 4096, "min": 0, "max": 16384, "step": 256,
                    "tooltip": "Token/sequence dimension above which to apply chunking.",
                }),
            }
        }

    RETURN_TYPES = ("LTX25_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "build"
    CATEGORY = "ltx25-int4-train"
    TITLE = "modify version from kjNodes ltx 2.5 Chunk FeedForward"

    def build(self, model, chunks, dim_threshold):
        # Return the SAME model, stamped with the feed-forward chunking config so the
        # Train node (which loads its own model from run.json) knows to chunk the FFN.
        out = dict(model or {})
        out["ffn_chunks"] = max(1, int(chunks))
        out["ffn_dim_threshold"] = max(0, int(dim_threshold))
        print(f"[O2noorLTX25ChunkFeedForward] chunks={out['ffn_chunks']} "
              f"dim_threshold={out['ffn_dim_threshold']} (model->model)", flush=True)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "O2noorLTX25ChunkFeedForward": O2noorLTX25ChunkFeedForward,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "O2noorLTX25ChunkFeedForward": "modify version from kjNodes ltx 2.5 Chunk FeedForward",
}
