// LTX-2.5 Int4 — LIVE Logs widget (step-by-step loss table).
//
// Adds a live-updating DOM widget to the Logs node that polls the run's telemetry
// every second and renders a clean per-step table:
//     step 12 | loss 0.982 | 0.26 s/step | ETA 24:10 | 6.9 GB
// with the newest step on top. Uses the modern addDOMWidget contract.
//
// The telemetry path comes from (in priority order): the Train node's websocket
// announcement (set at run start), the node's own INPUT data after it executes,
// or a Train node connected in the graph.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Reload-safe path recovery: the telemetry path is only known at runtime
// (websocket event / onExecuted), which is lost when switching workflows. We
// re-derive it from the Train node's serialized run_name + ComfyUI's output dir.
let OUTPUT_DIR_CACHE = null;

function findTrainNode(node) {
  if (!node.graph || !node.graph._nodes) return null;
  for (const o of node.graph._nodes) {
    if (o.type === "O2noorLTX25Int4Train") return o;
  }
  return null;
}

function readWidget(node, name) {
  if (!node || !node.widgets) return null;
  for (const w of node.widgets) if (w.name === name) return w.value;
  return null;
}

function normalizePath(p) {
  return p ? String(p).replace(/\\/g, "/") : p;
}

async function fetchOutputDir() {
  if (OUTPUT_DIR_CACHE) return OUTPUT_DIR_CACHE;
  try {
    const resp = await api.fetchApi("/ltx25/output_dir");
    const d = await resp.json();
    if (d && d.ok && d.output_dir) OUTPUT_DIR_CACHE = d.output_dir;
    return OUTPUT_DIR_CACHE;
  } catch (e) {
    return null;
  }
}

async function deriveTelPath(node) {
  // Find the LATEST run matching the Train node's run_name (incl. auto_unique suffix).
  const train = findTrainNode(node);
  const runName = train ? readWidget(train, "run_name") : null;
  if (!runName) return null;
  try {
    const resp = await api.fetchApi(`/ltx25/latest_run?name=${encodeURIComponent(runName)}`);
    const d = await resp.json();
    if (d && d.ok && d.telemetry_path) {
      const t = await api.fetchApi(`/ltx25/telemetry?path=${encodeURIComponent(d.telemetry_path)}`);
      const td = await t.json();
      if (td && td.ok && td.lines) return d.telemetry_path;
    }
  } catch (e) { /* ignore */ }
  return null;
}

function fmtLine(l) {
  let e;
  try {
    e = JSON.parse(l);
  } catch {
    return l;
  }
  if (!e || typeof e !== "object") return l;
  if (e.event === "step") {
    const sps = e.steps_per_sec || 0;
    const et = Number.isFinite(e.eta_s) ? e.eta_s : NaN;
    let eta = "—";
    if (Number.isFinite(et)) {
      const m = Math.floor(et / 60);
      const s = Math.round(et % 60);
      eta = `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    }
    // Fixed-width columns (monospace) so video + image lines stay aligned.
    const padR = (s, n) => String(s).padEnd(n);
    const padL = (s, n) => String(s).padStart(n);
    const loss = Number.isFinite(e.loss) ? e.loss.toFixed(4) : "nan";
    // Always render the full breakdown with a fixed-width audio slot.
    const vv = Number.isFinite(e.loss_video) ? e.loss_video.toFixed(4) : "----";
    const aa = Number.isFinite(e.loss_audio) ? e.loss_audio.toFixed(4) : "------";
    const lossExt = `(v${vv} a${aa})`.padEnd(19);
    const st = Number.isFinite(e.step_time) ? e.step_time.toFixed(2) : "0.00";
    let vram = "?";
    if (e.peak_vram && typeof e.peak_vram === "object") {
      const parts = Object.entries(e.peak_vram).map(([g, v]) => `${g}:${Number(v).toFixed(1)}`);
      const total = Number.isFinite(e.peak_vram_gb) ? ` \u00b7 ${e.peak_vram_gb.toFixed(1)}GB` : "";
      vram = parts.join(" ") + total;
    } else if (Number.isFinite(e.peak_vram_gb)) {
      vram = `${e.peak_vram_gb.toFixed(1)}GB`;
    }
    return `step ${padL(e.step, 4)} | loss ${padL(loss, 6)} ${lossExt}| ${padL(st, 6)} s/step | ${padL(sps.toFixed(2), 5)} step/s | ETA ${padL(eta, 5)} | VRAM ${vram}`;
  }
  if (e.event === "start") return `[start] ${e.run_id}  world=${e.world}  steps=${e.steps}`;
  if (e.event === "done") return `[done] ${e.total_steps} steps in ${e.elapsed_s}s`;
  if (e.event === "checkpoint") return `[ckpt] step ${e.step} -> ${e.path}`;
  return l;
}

const LOGS_EXT = {
  name: "ltx25-live-logs",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData || nodeData.name !== "O2noorLTX25Int4LogsOutputs") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    const onExecuted = nodeType.prototype.onExecuted;

    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const node = this;
      try {
        if ((node.widgets || []).some((w) => w.name === "ltx25-live-log")) return r;

        const wrap = document.createElement("div");
        wrap.style.cssText = "width:100%;box-sizing:border-box;background:#101820;display:flex;flex-direction:column;gap:4px;";
        const header = document.createElement("div");
        header.style.cssText = "display:flex;align-items:center;gap:6px;";
        const clearBtn = document.createElement("button");
        clearBtn.textContent = "Clear";
        clearBtn.title = "Clear the current run's telemetry (reset to waiting for a run)";
        clearBtn.style.cssText =
          "cursor:pointer;background:#3a1d1d;color:#f0a35e;border:1px solid #5a2a2a;" +
          "border-radius:4px;padding:2px 8px;font-size:11px;font-weight:bold;";
        header.appendChild(clearBtn);
        const hdrLabel = document.createElement("span");
        hdrLabel.style.cssText = "color:#8ff08a;font:11px monospace;opacity:.8;";
        hdrLabel.textContent = "Live training log";
        header.appendChild(hdrLabel);
        const clearStatus = document.createElement("span");
        clearStatus.style.cssText = "color:#8ff08a;font:11px monospace;opacity:.9;";
        header.appendChild(clearStatus);
        wrap.appendChild(header);

        const box = document.createElement("div");
        box.style.cssText =
          "width:100%;box-sizing:border-box;background:#101820;color:#8ff08a;" +
          "font:12px/16px monospace;padding:6px;white-space:pre-wrap;overflow-y:auto;flex:1;";
        box.textContent = "Waiting for run…";
        wrap.appendChild(box);

        const linesWidget = (node.widgets || []).find((w) => w.name === "lines");
        const getLines = () => (linesWidget ? (Number(linesWidget.value) || 100) : 100);

        const widget = node.addDOMWidget("ltx25-live-log", "ltx25-live-log", wrap, {
          getValue: () => box.textContent,
          setValue: (v) => {
            box.textContent = v || "";
          },
          serializeValue: () => "",
        });
        widget.serialize = false;
        widget.computeLayoutSize = () => {
          const w = Math.max(460, (node.size && node.size[0]) || 460);
          const h = Math.max(200, ((node.size && node.size[1]) || 300) - 66);
          return { minHeight: h, maxHeight: h, minWidth: w, maxWidth: 1e6 };
        };

        let telPath = null;
        let lastKey = "";

        const grabPath = () => {
          try {
            const d = node.getInputData ? node.getInputData(0) : null;
            if (d && d.telemetry_path) telPath = d.telemetry_path;
            if (node.telemetry_path) telPath = node.telemetry_path;
          } catch (e) {
            /* ignore */
          }
          if (!telPath && node.graph && node.graph._nodes) {
            try {
              const inLink = node.inputs && node.inputs[0] ? node.inputs[0].link : null;
              for (const o of node.graph._nodes) {
                if (o.type === "O2noorLTX25Int4Train" && o.telemetry_path && o.outputs && o.outputs[0]) {
                  const links = o.outputs[0].links || [];
                  if (inLink == null || links.includes(inLink)) {
                    telPath = o.telemetry_path;
                    break;
                  }
                }
              }
            } catch (e) {
              /* ignore */
            }
          }
          return telPath;
        };

        clearBtn.addEventListener("click", async () => {
          let ok = false;
          try {
            const p = grabPath() || (await deriveTelPath(node));
            if (p) {
              const resp = await api.fetchApi(`/ltx25/clear_run?path=${encodeURIComponent(p)}`, { method: "POST" });
              const d = await resp.json();
              ok = !!(d && d.ok);
            }
          } catch (e) {
            ok = false;
          }
          box.textContent = "Waiting for run…";
          lastKey = "";
          node.graph?.setDirtyCanvas(true, true);
          clearStatus.textContent = ok ? "cleared" : "failed — restart ComfyUI";
          clearStatus.style.color = ok ? "#8ff08a" : "#f0a35e";
          setTimeout(() => { clearStatus.textContent = ""; }, 4000);
        });

        const render = (lines) => {
          const n = getLines();
          const newestFirst = lines.slice(-n).reverse().join("\n");
          if (newestFirst !== lastKey) {
            lastKey = newestFirst;
            box.textContent = newestFirst;
            node.graph?.setDirtyCanvas(true, true);
          }
        };

        // Train node announces telemetry_path at run start -> live during the run.
        const onTelemetry = (event) => {
          try {
            const d = event && event.detail;
            if (d && d.telemetry_path) {
              telPath = d.telemetry_path;
              if (node.graph && node.graph._nodes) {
                for (const o of node.graph._nodes) {
                  if (o.type === "O2noorLTX25Int4Train") o.telemetry_path = d.telemetry_path;
                }
              }
            }
          } catch (e) {
            /* ignore */
          }
        };
        api.addEventListener("ltx25:telemetry", onTelemetry);

        const timer = setInterval(async () => {
          let path = grabPath();
          if (!path) path = await deriveTelPath(node);
          if (!path) return;
          try {
            const resp = await api.fetchApi(`/ltx25/telemetry?path=${encodeURIComponent(path)}`);
            const data = await resp.json();
            if (data && data.ok && data.lines?.length) {
              render(data.lines.map(fmtLine));
            }
          } catch (e) {
            /* transient */
          }
        }, 1000);

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
          clearInterval(timer);
          api.removeEventListener("ltx25:telemetry", onTelemetry);
          onRemoved?.apply(this, arguments);
        };
      } catch (e) {
        console.error("[ltx25-live-logs] setup error:", e);
      }
      return r;
    };

    // After the Logs node itself executes, also store the path from its inputs.
    nodeType.prototype.onExecuted = function (message) {
      const res = onExecuted?.apply(this, arguments);
      try {
        const inputs = (message && message.input) || {};
        if (inputs && inputs.run && inputs.run.telemetry_path) {
          this.telemetry_path = inputs.run.telemetry_path;
        }
      } catch (e) {
        /* ignore */
      }
      return res;
    };
  },
};

app.registerExtension(LOGS_EXT);
