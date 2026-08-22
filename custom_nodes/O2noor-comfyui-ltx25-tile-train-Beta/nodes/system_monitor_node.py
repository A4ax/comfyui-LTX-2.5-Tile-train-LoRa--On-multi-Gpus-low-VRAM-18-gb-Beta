"""O2noorLTX25Int4SystemMonitor — live system monitor (all GPUs + RAM + CPU).

Takes the Train node's `run` output and renders a live dashboard of the machine's
real-time resource usage: every GPU (memory used/total, utilization, temperature),
system RAM, and CPU. The web widget (system_monitor.js) polls the /ltx25/system
endpoint every second so it moves live. OUTPUT display node (no model output).
"""


class O2noorLTX25Int4SystemMonitor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "run": ("LTX25_RUN", {"tooltip": "The run output from LTX 2.5 Int4 Train (wiring only)."}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "ltx25-int4-train"
    TITLE = "O2noor LTX 2.5 System Monitor"
    OUTPUT_NODE = True

    def noop(self, run):
        return ()


NODE_CLASS_MAPPINGS = {
    "O2noorLTX25Int4SystemMonitor": O2noorLTX25Int4SystemMonitor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "O2noorLTX25Int4SystemMonitor": "O2noor LTX 2.5 System Monitor",
}
