"""O2noorLTX25Int4Metrics — rich live metrics dashboard node.

Takes the Train node's `run` output and renders a live dashboard (circular
progress ring, loss/video/audio loss, s/step, step/s, ETA, per-GPU VRAM bars,
grads, and a collapsible loss history chart). The web widget
(metrics_dashboard.js) polls the run's telemetry every second so it moves live
during training. OUTPUT display node (no model output).
"""


class O2noorLTX25Int4Metrics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "run": ("LTX25_RUN", {"tooltip": "The run output from LTX 2.5 Int4 Train."}),
            },
            "optional": {
                "steps_window": ("INT", {"default": 200, "min": 10, "max": 100000, "step": 10,
                                         "tooltip": "How many recent steps the history chart shows."}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "ltx25-int4-train"
    TITLE = "O2noor LTX 2.5 Metrics Dashboard"
    OUTPUT_NODE = True

    def noop(self, run, steps_window=200):
        return ()


NODE_CLASS_MAPPINGS = {
    "O2noorLTX25Int4Metrics": O2noorLTX25Int4Metrics,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "O2noorLTX25Int4Metrics": "O2noor LTX 2.5 Metrics Dashboard",
}
