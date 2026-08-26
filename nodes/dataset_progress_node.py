"""O2noorLTX25Int4DatasetProgress — live dataset-build progress monitor.

A display node (no output) with a web widget that streams what the Voice Dataset
node is doing in real time (module loading, video cutting, image conversion,
encoding) into ComfyUI — instead of the dataset node appearing frozen.

It REQUIRES the Voice Dataset node's `LTX25_DATASET` output (so it is part of the
workflow), and has a configurable `lines` count. The web widget polls the
/ltx25/dataset_progress endpoint independently, so it updates live while the
dataset node is still processing.
"""


class O2noorLTX25Int4DatasetProgress:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("LTX25_DATASET", {
                    "tooltip": "The dataset output from O2noor LTX 2.5 Voice Dataset (must be connected)."}),
                "lines": ("INT", {"default": 400, "min": 10, "max": 5000, "step": 10,
                                  "tooltip": "How many recent log lines to show (auto-scroll)."}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "ltx25-int4-train"
    TITLE = "O2noor LTX 2.5 Dataset Progress"
    OUTPUT_NODE = True

    def noop(self, dataset=None, lines=400):
        return ()


NODE_CLASS_MAPPINGS = {
    "O2noorLTX25Int4DatasetProgress": O2noorLTX25Int4DatasetProgress,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "O2noorLTX25Int4DatasetProgress": "O2noor LTX 2.5 Dataset Progress",
}
