"""O2noorLTX25Int4PreviewDataset — show every image the LoRA learns from as a visible gallery.

Takes the Dataset output and renders the actual training images as IMAGE tensors
so they appear on the canvas (no guessing what the model sees). Also reports how
many images/videos the dataset contains.
"""
import math
import os
import uuid

import numpy as np
import torch
from PIL import Image

from .. import pack_config


class O2noorLTX25Int4PreviewDataset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("LTX25_DATASET", {"tooltip": "The dataset from O2noorLTX25Int4Dataset."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview", "info")
    FUNCTION = "preview"
    CATEGORY = "ltx25-int4-train"
    TITLE = "O2noor LTX 2.5 Int4 Preview Dataset"
    OUTPUT_NODE = True

    def preview(self, dataset):
        tensor, info = self._build_preview(dataset)
        print(f"[O2noorLTX25Int4PreviewDataset] {info}", flush=True)
        # Save a real tiled preview image to the temp dir and reference it so the
        # frontend can actually load it (the old "preview" reference was never a
        # real file -> "Image failed to load").
        filename = self._save_preview(tensor)
        return {"ui": {"images": [{"filename": filename, "subfolder": "", "type": "temp"}]},
                "result": (tensor, info)}

    def _save_preview(self, tensor):
        import folder_paths
        tmp = folder_paths.get_temp_directory()
        os.makedirs(tmp, exist_ok=True)
        filename = f"ltx25_preview_{uuid.uuid4().hex[:12]}.png"
        tensor = tensor.detach().cpu()
        B = tensor.shape[0]
        H, W = tensor.shape[1], tensor.shape[2]
        cols = max(1, int(math.ceil(math.sqrt(B))))
        rows = max(1, int(math.ceil(B / cols)))
        grid = Image.new("RGB", (W * cols, H * rows), (0, 0, 0))
        for i in range(B):
            arr = (tensor[i].clamp(0, 1).numpy() * 255).astype(np.uint8)
            grid.paste(Image.fromarray(arr, "RGB"), ((i % cols) * W, (i // cols) * H))
        grid.save(os.path.join(tmp, filename))
        return filename

    def _build_preview(self, dataset):
        root = dataset.get("dataset_root", "")
        # image paths stored by the Dataset node (uploaded faces)
        img_paths = dataset.get("images") or []
        media_root = pack_config.media_upload_dir()
        resolved = []
        for p in img_paths:
            full = os.path.join(media_root, p) if not os.path.isabs(p) else p
            if os.path.exists(full):
                resolved.append(full)

        # fallback: scan the dataset dir for any image/clip sources
        if not resolved and root:
            for base, _, files in os.walk(root):
                for fn in sorted(files):
                    if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                        resolved.append(os.path.join(base, fn))

        images = []
        for p in resolved[:48]:  # cap display at 48
            try:
                img = Image.open(p).convert("RGB")
                img.thumbnail((256, 256))
                t = torch.from_numpy(np.asarray(img).copy()).float() / 255.0  # H,W,3
                images.append(t)
            except Exception:
                continue

        if images:
            H = max(i.shape[0] for i in images)
            W = max(i.shape[1] for i in images)
            batch = []
            for t in images:
                # center-pad to common size
                ph, pw = t.shape[:2]
                padded = torch.zeros((H, W, 3))
                y0, x0 = (H - ph) // 2, (W - pw) // 2
                padded[y0:y0 + ph, x0:x0 + pw] = t
                batch.append(padded)
            tensor = torch.stack(batch, dim=0)  # B,H,W,3
        else:
            tensor = torch.zeros((1, 256, 256, 3))

        info = (f"{len(img_paths)} image(s) uploaded, {dataset.get('samples', 0)} training clip(s)")
        return (tensor, info)