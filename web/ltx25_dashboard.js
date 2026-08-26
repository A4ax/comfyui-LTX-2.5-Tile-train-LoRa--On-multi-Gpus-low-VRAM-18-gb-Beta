// LTX-2.5 Int4 — LIVE Progress dashboard + Summary viewer.
//
// Adds two modern DOM widgets:
//   1. O2noorLTX25Int4Progress  -> circular progress ring (0-100%), live ETA, loss line chart.
//   2. O2noorLTX25Int4SummaryViewer -> run summary / checkpoint / directory info.
//
// The Progress widget polls the run's telemetry every second (same mechanism as
// the live-logs widget) so the ring / ETA / chart move live during training.
//
// Reload-safety: the telemetry path is only known at runtime (websocket event +
// onExecuted), which is lost when you switch workflows and come back. To survive
// that, we re-derive the path from the Train node's serialized `run_name` widget
// plus ComfyUI's output dir, and read the run total from each step's
// `total_steps` (or fall back to run.json).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

async function fetchTelemetry(path) {
  try {
    const resp = await api.fetchApi(`/ltx25/telemetry?path=${encodeURIComponent(path)}`);
    const data = await resp.json();
    if (!data || !data.ok || !data.lines) return [];
    return data.lines.map((l) => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);
  } catch (e) {
    return [];
  }
}

function formatEta(etaSec) {
  if (!Number.isFinite(etaSec) || etaSec < 0) return "—";
  const d = Math.floor(etaSec / 86400);
  const h = Math.floor((etaSec % 86400) / 3600);
  const m = Math.floor((etaSec % 3600) / 60);
  const s = Math.round(etaSec % 60);
  if (d > 0) return `${d}d ${h}h ${m}m ${s}s`;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

// ---- Reload-safe path helpers -------------------------------------------------
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

// Derive telemetry_path from the Train node's serialized run_name + output dir.
// Only returns a path that actually exists (so "waiting for a run" stays honest).
async function deriveTelPath(node) {
  // Find the LATEST run whose name matches the Train node's run_name (or the
  // auto_unique timestamp suffix of it), so after a reload we show the newest run.
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

// Read the run total (steps) from run.json next to the telemetry file.
async function fetchRunTotal(telPath) {
  if (!telPath) return 0;
  try {
    const resp = await api.fetchApi(`/ltx25/runinfo?path=${encodeURIComponent(telPath)}`);
    const d = await resp.json();
    if (d && d.ok && d.run && d.run.steps) return Number(d.run.steps) || 0;
  } catch (e) { /* ignore */ }
  return 0;
}

function grabTelPath(node) {
  if (node.telemetry_path) return node.telemetry_path;
  if (node.graph && node.graph._nodes) {
    for (const o of node.graph._nodes) {
      if (o.type === "O2noorLTX25Int4Train" && o.telemetry_path) return o.telemetry_path;
    }
  }
  return null;
}

const DASH_EXT = {
  name: "ltx25-dashboard",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData) return;
    if (nodeData.name === "O2noorLTX25Int4Progress") addProgressWidget(nodeType);
    else if (nodeData.name === "O2noorLTX25Int4SummaryViewer") addSummaryWidget(nodeType);
  },
};

function addProgressWidget(nodeType) {
  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const r = onNodeCreated?.apply(this, arguments);
    const node = this;
    try {
      if ((node.widgets || []).some((w) => w.name === "ltx25-progress")) return r;

      const winWidget = (node.widgets || []).find((w) => w.name === "steps_window");
      const getWindow = () => (winWidget ? (Number(winWidget.value) || 200) : 200);

      const root = document.createElement("div");
      root.style.cssText =
        "width:100%;box-sizing:border-box;background:#0d1117;color:#e6edf3;" +
        "font-family:Inter,system-ui,sans-serif;padding:10px;border-radius:8px;display:flex;flex-direction:column;gap:8px;";

      const top = document.createElement("div");
      top.style.cssText = "display:flex;align-items:center;gap:14px;";
      const NS = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("width", "120");
      svg.setAttribute("height", "120");
      svg.setAttribute("viewBox", "0 0 120 120");
      const R = 52, C = 2 * Math.PI * R;
      const bg = document.createElementNS(NS, "circle");
      bg.setAttribute("cx", "60"); bg.setAttribute("cy", "60"); bg.setAttribute("r", R);
      bg.setAttribute("fill", "none"); bg.setAttribute("stroke", "#1f2937"); bg.setAttribute("stroke-width", "10");
      const ring = document.createElementNS(NS, "circle");
      ring.setAttribute("cx", "60"); ring.setAttribute("cy", "60"); ring.setAttribute("r", R);
      ring.setAttribute("fill", "none"); ring.setAttribute("stroke", "#22d3ee");
      ring.setAttribute("stroke-width", "10"); ring.setAttribute("stroke-linecap", "round");
      ring.setAttribute("stroke-dasharray", C); ring.setAttribute("stroke-dashoffset", C);
      ring.style.transition = "stroke-dashoffset .4s ease";
      svg.appendChild(bg); svg.appendChild(ring);
      const pctText = document.createElementNS(NS, "text");
      pctText.setAttribute("x", "60"); pctText.setAttribute("y", "62");
      pctText.setAttribute("text-anchor", "middle"); pctText.setAttribute("dominant-baseline", "middle");
      pctText.setAttribute("fill", "#22d3ee");
      pctText.style.fontSize = "20px"; pctText.style.fontWeight = "700";
      pctText.textContent = "0%";
      svg.appendChild(pctText);
      top.appendChild(svg);

      const stats = document.createElement("div");
      stats.style.cssText = "display:flex;flex-direction:column;gap:6px;font-size:13px;";
      stats.innerHTML =
        '<div><span style="color:#8b949e">step</span> <b id="pr-step">0</b><span style="color:#8b949e"> / <span id="pr-total">0</span></b></div>' +
        '<div><span style="color:#8b949e">video</span> <b id="pr-loss" style="color:#22d3ee">—</b></div>' +
        '<div><span style="color:#8b949e">audio</span> <b id="pr-loss-audio" style="color:#f0a35e">—</b></div>' +
        '<div><span style="color:#8b949e">ETA</span> <b id="pr-eta">—</b></div>' +
        '<div><span style="color:#8b949e">rate</span> <b id="pr-rate">—</b></div>';
      top.appendChild(stats);

      const clearBtn = document.createElement("button");
      clearBtn.textContent = "Clear";
      clearBtn.title = "Clear the current run's telemetry (reset to waiting for a run)";
      clearBtn.style.cssText =
        "align-self:flex-start;cursor:pointer;background:#3a1d1d;color:#f0a35e;" +
        "border:1px solid #5a2a2a;border-radius:4px;padding:3px 9px;font-size:11px;font-weight:bold;";
      clearBtn.addEventListener("click", async () => {
        let ok = false;
        try {
          const p = telPath || grabTelPath(node) || (await deriveTelPath(node));
          if (p) {
            const resp = await api.fetchApi(`/ltx25/clear_run?path=${encodeURIComponent(p)}`, { method: "POST" });
            const d = await resp.json();
            ok = !!(d && d.ok);
          }
        } catch (e) { ok = false; }
        lastPct = -1;
        pctText.textContent = "0%";
        ring.setAttribute("stroke-dashoffset", C);
        root.querySelector("#pr-step").textContent = 0;
        root.querySelector("#pr-total").textContent = 0;
        root.querySelector("#pr-loss").textContent = "—";
        root.querySelector("#pr-loss-audio").textContent = "—";
        root.querySelector("#pr-eta").textContent = "—";
        root.querySelector("#pr-rate").textContent = "—";
        drawChart([], []);
        node.graph?.setDirtyCanvas(true, true);
        clearStatus.textContent = ok ? "cleared" : "failed — restart ComfyUI";
        clearStatus.style.color = ok ? "#8ff08a" : "#f0a35e";
        setTimeout(() => { clearStatus.textContent = ""; }, 4000);
      });
      top.appendChild(clearBtn);
      const clearStatus = document.createElement("span");
      clearStatus.style.cssText = "font-size:11px;color:#8ff08a;min-width:90px;";
      top.appendChild(clearStatus);
      root.appendChild(top);

      const canvas = document.createElement("canvas");
      canvas.style.cssText = "width:100%;height:120px;background:#0b0f14;border:1px solid #1f2937;border-radius:6px;display:block;";
      root.appendChild(canvas);
      const ctx = canvas.getContext("2d");

      const widget = node.addDOMWidget("ltx25-progress", "ltx25-progress", root, {
        getValue: () => "",
        setValue: () => {},
        serializeValue: () => "",
      });
      widget.serialize = false;
      // Height sized to fit the SVG row (120) + gap + chart (120) + padding.
      widget.computeLayoutSize = () => ({ minHeight: 120, maxHeight: 4000, minWidth: 300, maxWidth: 1e6 });

      let telPath = null;
      let lastPct = -1;
      let lastLosses = [], lastALosses = [];

      const LABEL_H = 18; // reserved bottom band for the min/max label
      const drawChart = (history, audioHistory) => {
        const w = canvas.width, h = canvas.height;
        ctx.clearRect(0, 0, w, h);
        const n = getWindow();
        const data = (history || []).slice(-n);
        const aData = (audioHistory || []).slice(-n);
        if (data.length < 2) {
          ctx.fillStyle = "#8b949e"; ctx.font = "12px Inter, system-ui";
          ctx.fillText("loss history…", 8, 16);
          return;
        }
        const all = data.concat(aData.filter((v) => v != null));
        const min = all.length ? Math.min(...all) : 0;
        const max = all.length ? Math.max(...all) : 1;
        const span = (max - min) || 1;
        const drawableH = Math.max(h - LABEL_H - 4, 10);
        const yFor = (v) => 4 + (1 - (v - min) / span) * (drawableH - 8);
        // blue: video loss
        ctx.strokeStyle = "#22d3ee"; ctx.lineWidth = 2; ctx.beginPath();
        data.forEach((v, i) => {
          const x = (i / (data.length - 1)) * w;
          const y = yFor(v);
          i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();
        // orange: audio loss, with CUT gaps where there's no audio (null).
        ctx.strokeStyle = "#f0a35e"; ctx.lineWidth = 1.5; ctx.beginPath();
        let pen = false;
        aData.forEach((v, i) => {
          if (v == null) { pen = false; return; }
          const x = (i / (aData.length - 1)) * w;
          const y = yFor(v);
          if (!pen) { ctx.moveTo(x, y); pen = true; }
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
        // Label drawn on top in its own filled band so the line never covers it.
        ctx.fillStyle = "#0b0f14";
        ctx.fillRect(0, h - LABEL_H, w, LABEL_H);
        ctx.fillStyle = "#8b949e"; ctx.font = "11px Inter, system-ui";
        ctx.fillText(`loss  min ${min.toFixed(3)}  max ${max.toFixed(3)}  (last ${data.length})   blue=video  orange=audio`, 6, h - 5);
      };

      const resize = () => {
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        if (rect.width > 0) {
          canvas.width = rect.width * dpr;
          canvas.height = rect.height * dpr;
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
      };

      const onTelemetry = (ev) => {
        try {
          const d = ev.detail;
          if (d && d.telemetry_path) {
            telPath = d.telemetry_path;
            if (node.graph) for (const o of node.graph._nodes) if (o.type === "O2noorLTX25Int4Train") o.telemetry_path = d.telemetry_path;
          }
        } catch (e) {}
      };
      api.addEventListener("ltx25:telemetry", onTelemetry);

      const timer = setInterval(async () => {
        try {
          if (!telPath) telPath = grabTelPath(node);
          if (!telPath) telPath = await deriveTelPath(node);
          if (!telPath) return;
          const evs = await fetchTelemetry(telPath);
          let total = 0, step = 0, loss = NaN, lossV = NaN, lossA = NaN, eta = NaN, rate = NaN;
          let done = false;
          const losses = [];
          const audioLosses = [];
          for (const e of evs) {
            if (e.event === "start") total = e.steps || 0;
            else if (e.event === "done") { done = true; total = e.total_steps || total; }
            else if (e.event === "step") {
              if (e.total_steps) total = e.total_steps;
              if (e.step >= step) step = e.step;
              if (Number.isFinite(e.loss_video)) lossV = e.loss_video;
              if (Number.isFinite(e.loss)) { loss = e.loss; losses.push(lossV); }
              audioLosses.push(Number.isFinite(e.loss_audio) ? e.loss_audio : null);
              if (Number.isFinite(e.loss_audio)) lossA = e.loss_audio;
              if (Number.isFinite(e.eta_s)) eta = e.eta_s;
              if (Number.isFinite(e.steps_per_sec)) rate = e.steps_per_sec;
            }
          }
          // Fallback: total may be missing from telemetry (older run / tail cut),
          // so read it from run.json next to the telemetry file.
          if (total <= 0) total = await fetchRunTotal(telPath);
          if (done) step = total;
          const pct = total > 0 ? (step / total) * 100 : 0;
          pctText.textContent = `${Math.round(pct)}%`;
          ring.setAttribute("stroke-dashoffset", C * (1 - pct / 100));
          root.querySelector("#pr-step").textContent = step;
          root.querySelector("#pr-total").textContent = total;
          root.querySelector("#pr-loss").textContent = Number.isFinite(lossV) ? lossV.toFixed(4) : "—";
          root.querySelector("#pr-loss-audio").textContent = Number.isFinite(lossA) ? lossA.toFixed(4) : "—";
          root.querySelector("#pr-eta").textContent = formatEta(eta);
          root.querySelector("#pr-rate").textContent = Number.isFinite(rate) ? `${rate.toFixed(2)} step/s` : "—";
          lastLosses = losses; lastALosses = audioLosses;
          drawChart(losses, audioLosses);
          if (Math.abs(pct - lastPct) > 0.05) { lastPct = pct; node.graph?.setDirtyCanvas(true, true); }
        } catch (e) {}
      }, 1000);

      const onRemoved = nodeType.prototype.onRemoved;
      nodeType.prototype.onRemoved = function () {
        clearInterval(timer);
        api.removeEventListener("ltx25:telemetry", onTelemetry);
        onRemoved?.apply(this, arguments);
      };

      setTimeout(resize, 120);
      window.addEventListener("resize", resize);
      // Re-render at full resolution when the node is enlarged (not stretched/blurry).
      if (typeof ResizeObserver !== "undefined") {
        new ResizeObserver(() => { resize(); drawChart(lastLosses, lastALosses); }).observe(root);
      }
    } catch (e) {
      console.error("[ltx25-progress] setup error:", e);
    }
    return r;
  };
}

function addSummaryWidget(nodeType) {
  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const r = onNodeCreated?.apply(this, arguments);
    const node = this;
    try {
      if ((node.widgets || []).some((w) => w.name === "ltx25-summary")) return r;

      const box = document.createElement("div");
      box.style.cssText =
        "width:100%;box-sizing:border-box;background:#0d1117;color:#e6edf3;" +
        "font:12px/16px ui-monospace,monospace;padding:8px;white-space:pre-wrap;overflow:hidden;border-radius:6px;";
      box.textContent = "Waiting for run info…";

      const widget = node.addDOMWidget("ltx25-summary", "ltx25-summary", box, {
        getValue: () => box.textContent,
        setValue: (v) => { box.textContent = v || ""; },
        serializeValue: () => "",
      });
      widget.serialize = false;
      widget.computeLayoutSize = () => ({ minHeight: 120, maxHeight: 4000, minWidth: 320, maxWidth: 1e6 });

      // Reload-safety: repopulate from run.json so switching workflows and coming
      // back doesn't leave the summary empty. RETRIES periodically (the graph / Train
      // node / run.json may not be ready at node-creation time), then stops once it
      // succeeds — same polling pattern as the Progress/Logs widgets.
      const populate = async () => {
        try {
          const train = findTrainNode(node);
          const runName = train ? readWidget(train, "run_name") : null;
          if (!runName) return false;
          // Resolve the ACTUAL run dir. With auto_unique ON the real folder has a
          // timestamp suffix (<name>_YYYYMMDD_HHMMSS), so joining the bare name
          // points at an old/empty folder. latest_run returns the newest one.
          let telPath = null;
          try {
            const lr = await api.fetchApi(`/ltx25/latest_run?name=${encodeURIComponent(runName)}`);
            const ld = await lr.json();
            if (ld && ld.ok && ld.telemetry_path) telPath = ld.telemetry_path;
          } catch (e) { /* ignore */ }
          if (!telPath) {
            const outDir = await fetchOutputDir();
            if (!outDir) return false;
            telPath = normalizePath(`${outDir}/${runName}/telemetry.jsonl`);
          }
          const resp = await api.fetchApi(`/ltx25/runinfo?path=${encodeURIComponent(telPath)}`);
          const d = await resp.json();
          if (!d || !d.ok || !d.run) return false;
          const rr = d.run;
          // telemetry -> status / final loss / VRAM / checkpoints / load times
          let done = false, gradsOk = true, finalLoss = NaN, finalV = NaN, finalA = NaN;
          let peakVRAM = 0, elapsed = 0, ckptCount = 0, ckptLast = "";
          const load = {};
          const evs = await fetchTelemetry(rr.telemetry_path || "");
          for (const e of evs) {
            if (e.event === "start") {
              load.model = e.model_load_s; load.lora = e.lora_setup_s; load.data = e.dataset_preload_s;
              load.opt = e.optimizer_s; load.total = e.total_init_s;
            } else if (e.event === "step") {
              if (Number.isFinite(e.loss)) finalLoss = e.loss;
              if (Number.isFinite(e.loss_video)) finalV = e.loss_video;
              if (Number.isFinite(e.loss_audio)) finalA = e.loss_audio;
              if (e.grads_finite === false) gradsOk = false;
              if (Number.isFinite(e.peak_vram_gb) && e.peak_vram_gb > peakVRAM) peakVRAM = e.peak_vram_gb;
            } else if (e.event === "checkpoint") { ckptCount++; ckptLast = e.path; }
            else if (e.event === "done") { done = true; elapsed = e.elapsed_s; }
          }
          const status = done ? (gradsOk ? "✅ completed · healthy" : "⚠️ completed · grads not finite") : "▶ running";
          const fmtS = (s) => (s ? Math.round(s) + "s" : "—");
          const lines = [
            `run      ${rr.run_id}`,
            `status   ${status}`,
            `steps    ${rr.steps}${elapsed ? "   time " + Math.round(elapsed) + "s" : ""}`,
            `lr       ${rr.lr}   rank ${rr.rank}  alpha ${rr.alpha}`,
            `bucket   ${(rr.bucket || []).join("x")}`,
            `tile     ${rr.tile}  overlap ${rr.overlap}   world ${rr.world}  blocks ${(rr.block_counts || []).join("/")}`,
          ];
          if (rr.mode || rr.train_audio) {
            const mode = rr.mode || (rr.train_audio ? "face+voice" : "face-only");
            lines.push(`mode     ${mode}${rr.segment_duration ? "  seg " + rr.segment_duration + "s" : ""}`);
          }
          if (Number.isFinite(finalLoss)) {
            let l = `loss     final ${finalLoss.toFixed(4)}`;
            if (Number.isFinite(finalV)) l += "  video " + finalV.toFixed(4);
            if (Number.isFinite(finalA)) l += "  audio " + finalA.toFixed(4);
            lines.push(l);
          }
          if (peakVRAM > 0) lines.push(`VRAM     peak ${peakVRAM.toFixed(1)} GB`);
          lines.push(`ckpt     ${ckptCount ? ckptCount + " saved" : "none"}`);
          if (ckptLast) lines.push(`         ${ckptLast}`);
          // Preprocessing model load times (text encoder, VAEs, embeddings processor).
          const preItems = [];
          try {
            const lt = await api.fetchApi(`/ltx25/load_times?path=${encodeURIComponent(rr.dataset_root || "")}`);
            const ld = await lt.json();
            if (ld && ld.ok && Array.isArray(ld.items)) for (const it of ld.items) if (it && it.component) preItems.push(it);
          } catch (e) { /* ignore */ }
          const loadParts = [];
          const order = ["text_encoder", "embeddings_processor", "video_vae", "audio_vae"];
          for (const c of order) { const it = preItems.find((x) => x.component === c); if (it) loadParts.push(`${c} ${fmtS(it.load_s)}`); }
          if (load.model) loadParts.push(`transformer ${fmtS(load.model)}`);
          if (loadParts.length) {
            let sum = 0; for (const it of preItems) if (Number.isFinite(it.load_s)) sum += it.load_s;
            if (Number.isFinite(load.total)) sum += load.total;
            lines.push(`load     ${loadParts.join("  ")}`);
            lines.push(`         total model load ~${Math.round(sum)}s`);
          }
          lines.push(`data     ${rr.dataset_root}`);
          box.textContent = lines.join("\n");
          return true;
        } catch (e) {
          return false;
        }
      };
      const summaryTimer = setInterval(async () => {
        if (await populate()) clearInterval(summaryTimer);
      }, 2000);
      populate(); // try immediately too

      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (msg) {
        const rr = onExecuted?.apply(this, arguments);
        try {
          const inp = (msg && msg.input) || {};
          const s = inp.summary;
          if (typeof s === "string" && s) box.textContent = s;
        } catch (e) {}
        return rr;
      };

      const onRemoved = nodeType.prototype.onRemoved;
      nodeType.prototype.onRemoved = function () {
        clearInterval(summaryTimer);
        onRemoved?.apply(this, arguments);
      };
    } catch (e) {
      console.error("[ltx25-summary] setup error:", e);
    }
    return r;
  };
}

app.registerExtension(DASH_EXT);
