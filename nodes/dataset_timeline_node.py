"""O2noorLTX25Int4DatasetTimeline — live dataset-pipeline timeline node.

Wire this to the `dataset` output of the Voice Dataset node (or any dataset
root). It renders a live, animated dashboard of everything the dataset encode is
doing inside so the node never looks frozen:

  - current stage + a moving spinner (cutting clips / audio extract / caption
    encode / precompute audio / VAE video+audio encode, or done),
  - ffmpeg clip-cutting: clips done / total, ~s/clip, elapsed+s (derived live
    from the scenes file writes),
  - per-model load times (from load_times.jsonl) with seconds + total,
  - live status event log tail (from status.jsonl),
  - GPU / RAM utilisation.

The web widget (dataset_timeline.js) polls /ltx25/dataset_timeline every second.
OUTPUT display node (no model output).
"""


class O2noorLTX25Int4DatasetTimeline:
    @classmethod
    def INPUT_TYPES(cls):
        from . import engine_driver
        try:
            from .. import pack_config
            default_path = pack_config.load_config().get("dataset_root") or engine_driver.engine_workdir()
        except Exception:
            default_path = engine_driver.engine_workdir()
        return {
            "required": {},
            "optional": {
                "dataset": ("LTX25_DATASET", {
                    "tooltip": "The dataset output from O2noor LTX 2.5 Voice Dataset. "
                               "Wiring only — used to find the dataset root to monitor.",
                }),
                "dataset_path": ("STRING", {
                    "default": default_path if default_path else "",
                    "tooltip": "The dataset root to monitor (defaults to the pack dataset_root). "
                               "Set this to the dataset output's path if you want a different one.",
                }),
                "poll_seconds": ("INT", {"default": 1, "min": 0.5, "max": 10.0, "step": 0.5,
                                         "tooltip": "How often the dashboard refreshes (seconds)."}),
                "show_gpu": ("BOOLEAN", {"default": True,
                                         "tooltip": "Show a live GPU/RAM strip too."}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "ltx25-int4-train"
    TITLE = "O2noor LTX 2.5 Dataset Timeline"
    OUTPUT_NODE = True

    def noop(self, dataset=None, dataset_path="", poll_seconds=1.0, show_gpu=True):
        return ()


NODE_CLASS_MAPPINGS = {
    "O2noorLTX25Int4DatasetTimeline": O2noorLTX25Int4DatasetTimeline,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "O2noorLTX25Int4DatasetTimeline": "O2noor LTX 2.5 Dataset Timeline",
}
