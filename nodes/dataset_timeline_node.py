"""O2noorLTX25Int4DatasetTimeline — live dataset-pipeline timeline node.

Wire this to the `dataset` output of the Voice Dataset node. It renders a live,
animated dashboard of everything the dataset encode is doing inside so the node
never looks frozen:

  - current stage + a spinner that only moves while a stage is actually running,
  - ffmpeg clip-cutting: clips done / total, ~clips/s, elapsed,
  - audio extract / caption / precompute / VAE video+audio encode bars,
  - per-model load times (from load_times.jsonl) with seconds + total,
  - live status event log tail (from status.jsonl).

NO HARDCoded PATHS: the node only monitors a dataset when it is WIRED to a
dataset (LTX25_DATASET) output. The dataset_root is resolved purely from the
wired dataset dict (dataset["dataset_root"]), never from config or a default.
Without a wire it shows an explicit "wire the dataset output" empty state — no
spinner, no misleading empty panel.

The web widget (dataset_timeline.js) polls /ltx25/dataset_timeline every second
using the dataset_root exposed by this node. OUTPUT display node (no model output).
"""


class O2noorLTX25Int4DatasetTimeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "dataset": ("LTX25_DATASET", {
                    "tooltip": "REQUIRED for monitoring: the dataset output from "
                               "O2noor LTX 2.5 Voice Dataset. The dataset root is "
                               "read from this connection — no path is guessed.",
                }),
                "poll_seconds": ("INT", {"default": 1, "min": 0.5, "max": 10.0, "step": 0.5,
                                         "tooltip": "How often the dashboard refreshes (seconds)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("dataset_root",)
    FUNCTION = "noop"
    CATEGORY = "ltx25-int4-train"
    TITLE = "O2noor LTX 2.5 Dataset Timeline"
    OUTPUT_NODE = True

    def noop(self, dataset=None, poll_seconds=1.0):
        # Resolve the monitored root ONLY from the wired dataset dict. If there is
        # no wire (or the dict carries no dataset_root), return "" so the widget
        # shows an explicit "wire the dataset output" empty state.
        root = ""
        if isinstance(dataset, dict):
            root = dataset.get("dataset_root") or ""
        return (root,)


NODE_CLASS_MAPPINGS = {
    "O2noorLTX25Int4DatasetTimeline": O2noorLTX25Int4DatasetTimeline,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "O2noorLTX25Int4DatasetTimeline": "O2noor LTX 2.5 Dataset Timeline",
}
