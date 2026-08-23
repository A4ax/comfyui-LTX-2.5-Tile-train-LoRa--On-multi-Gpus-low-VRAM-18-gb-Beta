"""LTX25 Batch Prompt node (enqueue driver).

Reads your multiline prompt list. The current prompt generates line 1 (via the
connected positive CLIPTextEncode). For each remaining line it enqueues a copy of
the current single-image workflow (this node removed) with the positive
CLIPTextEncode set to that line, so each image is generated + decoded + saved by
a full, normal ComfyUI prompt run. This is the mechanism that reliably saves
each image (ComfyUI's list-mapping does not decode/save).
"""

import copy

import requests
import server


class O2noorLTX25BatchPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_text": ("STRING", {"forceInput": True}),
                "save_prefix": ("STRING", {"default": "pose"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = "LTX25/Batch"
    TITLE = "O2noor LTX 2.5 Batch Prompt (100 poses)"

    def run(self, prompt_text, save_prefix="pose", prompt=None, unique_id=None, extra_pnginfo=None):
        lines = [ln.rstrip() for ln in prompt_text.split("\n") if ln.strip()]
        current = lines[0] if lines else ""
        if len(lines) < 2:
            return (current,)
        if not prompt:
            return (current,)

        base = {k: v for k, v in prompt.items() if k != str(unique_id)}
        target = self._find_positive_clip(base)
        save_node = self._find_save(base)
        if target is None or save_node is None:
            return (current,)

        port = getattr(server.PromptServer.instance, "port", 8188)
        url = f"http://127.0.0.1:{port}/prompt"
        client_id = (
            (extra_pnginfo or {}).get("client_id")
            or getattr(server.PromptServer.instance, "client_id", None)
        )

        for line in lines[1:]:
            p = copy.deepcopy(base)
            p[target]["inputs"]["text"] = line
            p[save_node]["inputs"]["filename_prefix"] = save_prefix
            try:
                requests.post(url, json={"prompt": p, "client_id": client_id}, timeout=120)
            except Exception as e:  # noqa: BLE001
                print(f"[O2noorLTX25BatchPrompt] enqueue error: {e}")
        return (current,)

    @staticmethod
    def _find_positive_clip(prompt):
        aid = next((nid for nid, nd in prompt.items()
                    if nd.get("class_type") == "ApplyInstantID"), None)
        if aid is not None:
            pos = prompt[aid]["inputs"].get("positive")
            if isinstance(pos, list) and len(pos) > 0:
                return str(pos[0])
        for nid, nd in prompt.items():
            if nd.get("class_type") == "CLIPTextEncode":
                return nid
        return None

    @staticmethod
    def _find_save(prompt):
        for nid, nd in prompt.items():
            if nd.get("class_type") == "SaveImage":
                return nid
        return None


NODE_CLASS_MAPPINGS = {"O2noorLTX25BatchPrompt": O2noorLTX25BatchPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"O2noorLTX25BatchPrompt": "O2noor LTX 2.5 Batch Prompt (100 poses)"}
