// LTX-2.5 Int4 Tile-Train — Dataset media widget (upload face images).
//
// ONE DOM widget (addDOMWidget) that draws an upload button + thumbnail gallery.
// Real HTML (native clicks) and deterministic height from content, so it renders
// correctly on the 1.49.6+ frontend. Face-only: no voice.
//
// Uploads POST to the pack's own /ltx25/upload/images endpoint, which saves into
// the ComfyUI input dir and returns { ok, files: [basenames] }. The underlying
// STRING store widget is hidden (computeSize -> [0,0]) and remains the source of
// truth for serialization.
import { app } from "../../scripts/app.js";

function parseList(v) {
  try {
    const a = JSON.parse(v || "[]");
    return Array.isArray(a) ? a : [];
  } catch {
    return [];
  }
}

async function uploadFiles(files, store, node, refresh) {
  const fd = new FormData();
  const arr = Array.from(files || []);
  if (!arr.length) return;
  for (const f of arr) fd.append("files", f);
  try {
    const resp = await fetch("/ltx25/upload/images", { method: "POST", body: fd });
    const data = await resp.json();
    const names = Array.isArray(data && data.files) ? data.files.filter(Boolean) : [];
    if (names.length) {
      store.value = JSON.stringify([...parseList(store.value), ...names]);
      store.callback?.(store.value);
      refresh();
    } else {
      console.error("[ltx25] upload returned no files:", data);
    }
  } catch (e) {
    console.error("[ltx25] upload failed:", e);
  }
}

function buildImageWidget(node, store) {
  const THUMB = 72;
  const CELL = THUMB + 8;

  const wrap = document.createElement("div");
  wrap.style.cssText =
    "width:100%;box-sizing:border-box;padding:2px 4px 4px;" +
    "font-family:sans-serif;user-select:none;overflow:hidden;";

  const btn = document.createElement("div");
  btn.setAttribute("data-ltx-upload", "images");
  btn.style.cssText =
    "box-sizing:border-box;width:100%;height:30px;line-height:26px;padding:0 10px;cursor:pointer;" +
    "border-radius:4px;background:#2a6a3e;color:#fff;font-weight:bold;font-size:13px;";
  const label = document.createElement("span");
  label.style.cssText = "float:right;font-weight:normal;opacity:.85;";
  btn.appendChild(document.createTextNode("Upload images"));
  btn.appendChild(label);

  const input = document.createElement("input");
  input.type = "file";
  input.multiple = true;
  input.accept = "image/*";
  input.style.display = "none";
  btn.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files && input.files.length) uploadFiles(input.files, store, node, refresh);
    input.value = "";
  });

  const gal = document.createElement("div");
  gal.style.cssText = "display:flex;flex-wrap:wrap;gap:6px;margin-top:6px;align-items:flex-start;" +
    "max-height:280px;overflow-y:auto;";
  wrap.append(btn, input, gal);

  // Fixed widget height: the gallery scrolls internally instead of making the node huge.
  const widgetHeight = () => 30 + 6 + 280 + 4;

  function render() {
    const list = parseList(store.value);
    label.textContent = `Uploaded images (${list.length})`;
    gal.replaceChildren();
    for (const p of list) {
      const cell = document.createElement("div");
      cell.style.cssText =
        `position:relative;width:${THUMB}px;height:${THUMB}px;background:#222;` +
        "border:1px solid #3a3a3a;border-radius:4px;overflow:hidden;";
      const img = document.createElement("img");
      img.style.cssText = "width:100%;height:100%;object-fit:cover;display:block;";
      img.src = `/view?filename=${encodeURIComponent(p)}&type=input&subfolder=`;
      img.onerror = () => {
        img.style.display = "none";
        const t = document.createElement("div");
        t.textContent = "image";
        t.style.cssText = "color:#888;font-size:11px;line-height:" + THUMB + "px;text-align:center;";
        cell.appendChild(t);
      };
      cell.appendChild(img);
      const rm = document.createElement("div");
      rm.textContent = "✕";
      rm.setAttribute("data-ltx-remove", "true");
      rm.style.cssText =
        "position:absolute;top:2px;right:2px;width:16px;height:16px;line-height:16px;text-align:center;" +
        "border-radius:50%;background:rgba(0,0,0,.7);color:#fff;font-size:11px;cursor:pointer;";
      rm.addEventListener("click", (e) => {
        e.stopPropagation();
        const idx = list.indexOf(p);
        if (idx >= 0) list.splice(idx, 1);
        store.value = JSON.stringify(list);
        store.callback?.(store.value);
        refresh();
      });
      cell.appendChild(rm);
      gal.appendChild(cell);
    }
    if (!list.length) {
      const empty = document.createElement("div");
      empty.textContent = "No images yet — click above to upload your faces";
      empty.style.cssText = "color:#999;font-size:12px;line-height:24px;";
      gal.appendChild(empty);
    }
  }

  function refresh() {
    render();
    try {
      node.setSize(node.computeSize());
    } catch (e) {
      /* not ready yet */
    }
    node.graph?.setDirtyCanvas(true, true);
  }

  render();

  const widget = node.addDOMWidget("ltx25-images-view", "ltx25-images", wrap, {
    getValue: () => store.value,
    setValue: (v) => {
      store.value = v;
      render();
    },
    serializeValue: () => store.value,
  });
  widget.serialize = false;
  widget.computeLayoutSize = () => {
    const h = widgetHeight();
    return { minHeight: h, maxHeight: h, minWidth: 300, maxWidth: 1e6 };
  };
  widget.render = render; // let the node re-render after a workflow is loaded
  widget.refresh = refresh;

  return widget;
}

const ext = {
  name: "ltx25-upload",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData || nodeData.name !== "O2noorLTX25Int4Dataset") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      try {
        const imgsStore = this.widgets.find((w) => w.name === "images");
        if (!imgsStore) return r;
        imgsStore.computeSize = () => [0, 0];
        imgsStore.type = "converted-widget";
        imgsStore.serialize = true;
        this._ltxMedia = buildImageWidget(this, imgsStore);
        if (this.size[0] < 300) this.size[0] = 300;
        this.setSize(this.computeSize());
        this.graph?.setDirtyCanvas(true, true);

        // Re-render the gallery after a workflow is loaded (ComfyUI sets the hidden
        // string widget's value directly, so render() isn't otherwise triggered).
        const refreshGallery = () => {
          try {
            this._ltxMedia?.render?.();
            this.setSize(this.computeSize());
            this.graph?.setDirtyCanvas(true, true);
          } catch (e) {
            /* ignore */
          }
        };
        const onConfigure = this.onConfigure;
        this.onConfigure = function () {
          const rc = onConfigure?.apply(this, arguments);
          refreshGallery();
          return rc;
        };
        setTimeout(refreshGallery, 300); // fallback: values may be applied shortly after creation
      } catch (e) {
        console.error("[ltx25] widget setup error:", e);
      }
      return r;
    };
  },
};

app.registerExtension(ext);
