"""LTX-2.5 Int4 live Progress + Summary display nodes.

O2noorLTX25Int4Progress:     live circular progress (0-100%), live ETA, and a loss line
                       chart. Web widget (ltx25_dashboard.js) polls the run's
                       telemetry every second so it moves live during training.
O2noorLTX25Int4SummaryViewer: shows the run summary / checkpoint / directory info.

Both are OUTPUT display nodes (no model output). Their web widgets are attached
by ltx25_dashboard.js.
"""


class O2noorLTX25Int4Progress:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "milestones": ("STRING", {"forceInput": True}),
                "steps_window": ("INT", {"default": 200, "min": 10, "max": 100000, "step": 10}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "ltx25-int4-train"
    TITLE = "O2noor LTX 2.5 Progress (Live)"
    OUTPUT_NODE = True

    def noop(self, milestones, steps_window):
        return ()


class O2noorLTX25Int4SummaryViewer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "summary": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "ltx25-int4-train"
    TITLE = "O2noor LTX 2.5 Summary (Info)"
    OUTPUT_NODE = True

    def noop(self, summary):
        return ()


NODE_CLASS_MAPPINGS = {
    "O2noorLTX25Int4Progress": O2noorLTX25Int4Progress,
    "O2noorLTX25Int4SummaryViewer": O2noorLTX25Int4SummaryViewer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "O2noorLTX25Int4Progress": "O2noor LTX 2.5 Progress (Live)",
    "O2noorLTX25Int4SummaryViewer": "O2noor LTX 2.5 Summary (Info)",
}
