// LTX-2.5 Voice Dataset — upload speaking VIDEOS (face+voice) and/or face IMAGES.
//
// Mirrors media_upload.js but for the O2noorLTX25Int4VoiceDataset node, with TWO widgets:
//   1. Speaking videos  -> thumbnail grid (first-frame .thumb.png generated at upload).
//   2. Face images      -> thumbnail grid (like the original face dataset node).
// Basenames are stored in the hidden "videos" / "images" STRING widgets.
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

function viewUrl(name) {
  return `/view?filename=${encodeURIComponent(name)}&type=input&subfolder=`;
}

function buildListWidget(node, store, opts) {
  const THUMB = 76;
  const CELL = THUMB + 34; // room for the filename label under each thumb

  const wrap = document.createElement("div");
  wrap.style.cssText =
    "width:100%;box-sizing:border-box;padding:2px 4px 4px;" +
    "font-family:sans-serif;user-select:none;overflow:hidden;";

  const btn = document.createElement("div");
  btn.style.cssText =
    "box-sizing:border-box;width:100%;height:28px;line-height:24px;padding:0 10px;cursor:pointer;" +
    `border-radius:4px;background:${opts.color};color:#fff;font-weight:bold;font-size:13px;`;
  const label = document.createElement("span");
  label.style.cssText = "float:right;font-weight:normal;opacity:.85;";
  btn.appendChild(document.createTextNode(opts.label));
  btn.appendChild(label);

  const input = document.createElement("input");
  input.type = "file";
  input.multiple = true;
  input.accept = opts.accept;
  input.style.display = "none";
  btn.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files && input.files.length) uploadFiles(input.files, store, node, refresh);
    input.value = "";
  });

  const gal = document.createElement("div");
  gal.style.cssText = "display:flex;flex-wrap:wrap;gap:6px;margin-top:6px;align-items:flex-start;" +
    "max-height:385px;overflow-y:auto;";
  wrap.append(btn, input, gal);

  // Fixed widget height: the gallery scrolls internally instead of making the node huge.
  const widgetHeight = () => 28 + 6 + 385 + 4;

  function render() {
    const list = parseList(store.value);
    label.textContent = `${opts.plural} (${list.length})`;
    gal.replaceChildren();
    if (!list.length) {
      const empty = document.createElement("div");
      empty.textContent = opts.emptyText;
      empty.style.cssText = "color:#999;font-size:12px;line-height:22px;";
      gal.appendChild(empty);
    }
    for (const p of list) {
      const cell = document.createElement("div");
      cell.style.cssText =
        `width:${THUMB}px;overflow:hidden;background:#1c1c1c;border:1px solid #333;` +
        "border-radius:4px;display:flex;flex-direction:column;";
      const thumbWrap = document.createElement("div");
      thumbWrap.style.cssText = `width:${THUMB}px;height:${THUMB}px;background:#222;position:relative;`;
      const img = document.createElement("img");
      img.style.cssText = "width:100%;height:100%;object-fit:cover;display:block;";
      img.src = opts.thumbFn ? opts.thumbFn(p) : viewUrl(p);
      img.onerror = () => {
        img.style.display = "none";
        const t = document.createElement("div");
        t.textContent = opts.icon || "•";
        t.style.cssText = "color:#888;font-size:22px;line-height:" + THUMB + "px;text-align:center;";
        thumbWrap.appendChild(t);
      };
      thumbWrap.appendChild(img);
      const rm = document.createElement("div");
      rm.textContent = "✕";
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
      thumbWrap.appendChild(rm);
      const nameEl = document.createElement("div");
      nameEl.textContent = p;
      nameEl.style.cssText = "color:" + opts.textColor + ";font-size:10px;padding:2px 4px;" +
        "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
      cell.append(thumbWrap, nameEl);
      gal.appendChild(cell);
    }
  }

  function refresh() {
    render();
    try {
      node.setSize(node.computeSize());
    } catch (e) {
      /* not ready */
    }
    node.graph?.setDirtyCanvas(true, true);
  }

  render();

  const widget = node.addDOMWidget(opts.domId, opts.domId, wrap, {
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
  widget.render = render;
  widget.refresh = refresh;

  return widget;
}

const ext = {
  name: "ltx25-voice-upload",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData || nodeData.name !== "O2noorLTX25Int4VoiceDataset") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      try {
        const videosStore = this.widgets.find((w) => w.name === "videos");
        const imagesStore = this.widgets.find((w) => w.name === "images");
        const refreshAll = () => {
          try {
            this._ltxVoiceVideos?.render?.();
            this._ltxVoiceImages?.render?.();
            this.setSize(this.computeSize());
            this.graph?.setDirtyCanvas(true, true);
          } catch (e) {
            /* ignore */
          }
        };
        if (videosStore) {
          videosStore.computeSize = () => [0, 0];
          videosStore.type = "converted-widget";
          videosStore.serialize = true;
          this._ltxVoiceVideos = buildListWidget(this, videosStore, {
            label: "Upload speaking videos",
            plural: "Speaking videos",
            accept: "video/*,.mp4,.mov,.mkv,.webm",
            color: "#6a4a2a",
            textColor: "#d7b88a",
            icon: "▶",
            emptyText: "No videos yet — add clips of the person speaking (auto-split into segments)",
            domId: "ltx25-videos-view",
            thumbFn: (p) => "/ltx25/thumb?file=" + encodeURIComponent(p),
          });
        }
        if (imagesStore) {
          imagesStore.computeSize = () => [0, 0];
          imagesStore.type = "converted-widget";
          imagesStore.serialize = true;
          this._ltxVoiceImages = buildListWidget(this, imagesStore, {
            label: "Upload face images",
            plural: "Face images",
            accept: "image/*",
            color: "#2a6a3e",
            textColor: "#9fd3ab",
            icon: "▨",
            emptyText: "Optional — add face images to reinforce identity (no voice)",
            domId: "ltx25-images-view",
          });
        }
        if (this.size[0] < 300) this.size[0] = 300;
        this.setSize(this.computeSize());
        this.graph?.setDirtyCanvas(true, true);

        const onConfigure = this.onConfigure;
        this.onConfigure = function () {
          const rc = onConfigure?.apply(this, arguments);
          refreshAll();
          return rc;
        };
        setTimeout(refreshAll, 300);
      } catch (e) {
        console.error("[ltx25] voice widget setup error:", e);
      }
      return r;
    };
  },
};

app.registerExtension(ext);
