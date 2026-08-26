// O2noor LTX 2.5 Dataset Progress - live dataset-build log widget.
//
// Shows what the Voice Dataset node is doing in real time (module loading,
// video cutting, image conversion, encoding) by polling /ltx25/dataset_progress.
// The widget polls independently of node execution, so it updates live while the
// dataset node is processing. `lines` on the node controls how many are shown.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
  name: "O2noor.LTX25.DatasetProgress",

  beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "O2noorLTX25Int4DatasetProgress") return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      try {
        if ((this.widgets || []).some((w) => w.name === "ltx25-dataset-progress")) return r;

        const root = document.createElement("div");
        root.style.cssText =
          "width:100%;box-sizing:border-box;background:#0d1117;color:#e6edf3;" +
          "font-family:ui-monospace,Consolas,monospace;font-size:11px;padding:8px;" +
          "border-radius:8px;border:1px solid #21262d;";

        const pre = document.createElement("pre");
        pre.style.cssText =
          "margin:0;max-height:320px;overflow:auto;white-space:pre-wrap;word-break:break-word;" +
          "color:#9aa7b8;line-height:1.45;";
        pre.textContent = "[dataset] waiting for dataset build...";
        root.appendChild(pre);

        const widget = this.addDOMWidget("ltx25-dataset-progress", "ltx25-dataset-progress", root, {
          getValue: () => "", setValue: () => {}, serializeValue: () => "",
        });
        widget.serialize = false;
        widget.computeLayoutSize = () => ({ minHeight: 120, maxHeight: 340, minWidth: 320, maxWidth: 1e6 });

        const linesWidget = (this.widgets || []).find((w) => w.name === "lines");
        const getLines = () => {
          const n = linesWidget ? Number(linesWidget.value) : 400;
          return Number.isFinite(n) && n > 0 ? n : 400;
        };
        let lastText = "";
        let lastHash = "";

        const tick = async () => {
          try {
            const resp = await api.fetchApi(`/ltx25/dataset_progress?lines=${getLines()}`);
            const data = await resp.json();
            if (!data || !data.ok) {
              if (lastText !== "[dataset] no active dataset build...") {
                lastText = "[dataset] no active dataset build...";
                pre.textContent = lastText;
              }
              return;
            }
            const text = data.lines.join("\n");
            if (text !== lastHash) {
              lastHash = text;
              pre.textContent = text;
              pre.scrollTop = pre.scrollHeight;
            }
          } catch (e) { /* ignore */ }
        };

        tick();
        const timer = setInterval(tick, 1000);
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
          clearInterval(timer);
          onRemoved?.apply(this, arguments);
        };
      } catch (e) {
        console.error("[ltx25-dataset-progress] setup error:", e);
      }
      return r;
    };
  },
});
