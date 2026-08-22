// LTX-2.5 Metrics Dashboard — rich live telemetry dashboard.
//
// One DOM widget on the O2noorLTX25Int4Metrics node. Polls the run's telemetry every
// second and renders:
//   - a circular progress ring (step %),
//   - metric tiles: loss / video loss / audio loss / s/step / step/s / ETA /
//     VRAM gpu0 / VRAM gpu1 / VRAM total / grads,
//   - per-GPU VRAM mini bars,
//   - a collapsible loss history chart (hidden by default).
// The telemetry path comes from the connected `run` input first, then the
// ltx25:telemetry event, then run_name + output dir (reload-safe).
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

function fmtEta(sec) {
  if (!Number.isFinite(sec) || sec < 0) return "—";
  const d = Math.floor(sec / 86400), h = Math.floor((sec % 86400) / 3600);
  const m = Math.floor((sec % 3600) / 60), s = Math.round(sec % 60);
  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

let OUTPUT_DIR_CACHE = null;
function findTrainNode(node) {
  if (!node.graph || !node.graph._nodes) return null;
  for (const o of node.graph._nodes) if (o.type === "O2noorLTX25Int4Train") return o;
  return null;
}
function readWidget(node, name) {
  if (!node || !node.widgets) return null;
  for (const w of node.widgets) if (w.name === name) return w.value;
  return null;
}
function normalizePath(p) { return p ? String(p).replace(/\\/g, "/") : p; }
async function fetchOutputDir() {
  if (OUTPUT_DIR_CACHE) return OUTPUT_DIR_CACHE;
  try {
    const resp = await api.fetchApi("/ltx25/output_dir");
    const d = await resp.json();
    if (d && d.ok && d.output_dir) OUTPUT_DIR_CACHE = d.output_dir;
    return OUTPUT_DIR_CACHE;
  } catch (e) { return null; }
}
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
  try { const d = node.getInputData ? node.getInputData(0) : null; if (d && d.telemetry_path) return d.telemetry_path; } catch (e) { /* ignore */ }
  if (node.telemetry_path) return node.telemetry_path;
  if (node.graph && node.graph._nodes) {
    for (const o of node.graph._nodes) if (o.type === "O2noorLTX25Int4Train" && o.telemetry_path) return o.telemetry_path;
  }
  return null;
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

const EXT = {
  name: "ltx25-metrics",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData || nodeData.name !== "O2noorLTX25Int4Metrics") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const node = this;
      try {
        if ((node.widgets || []).some((w) => w.name === "ltx25-metrics")) return r;

        const winWidget = (node.widgets || []).find((w) => w.name === "steps_window");
        const getWindow = () => (winWidget ? (Number(winWidget.value) || 200) : 200);

        const root = document.createElement("div");
        root.style.cssText =
          "width:100%;box-sizing:border-box;background:#0d1117;color:#e6edf3;" +
          "font-family:Inter,system-ui,sans-serif;padding:10px;border-radius:8px;";

        // ---- header: ring + summary ----
        const header = document.createElement("div");
        header.style.cssText = "display:flex;align-items:center;gap:14px;";
        const NS = "http://www.w3.org/2000/svg";
        const svg = document.createElementNS(NS, "svg");
        svg.setAttribute("width", "104"); svg.setAttribute("height", "104"); svg.setAttribute("viewBox", "0 0 104 104");
        const R = 46, C = 2 * Math.PI * R;
        const bg = document.createElementNS(NS, "circle");
        bg.setAttribute("cx", "52"); bg.setAttribute("cy", "52"); bg.setAttribute("r", R);
        bg.setAttribute("fill", "none"); bg.setAttribute("stroke", "#1f2937"); bg.setAttribute("stroke-width", "9");
        const ring = document.createElementNS(NS, "circle");
        ring.setAttribute("cx", "52"); ring.setAttribute("cy", "52"); ring.setAttribute("r", R);
        ring.setAttribute("fill", "none"); ring.setAttribute("stroke", "#22d3ee");
        ring.setAttribute("stroke-width", "9"); ring.setAttribute("stroke-linecap", "round");
        ring.setAttribute("stroke-dasharray", C); ring.setAttribute("stroke-dashoffset", C);
        ring.style.transition = "stroke-dashoffset .4s ease";
        svg.appendChild(bg); svg.appendChild(ring);
        const pct = document.createElementNS(NS, "text");
        pct.setAttribute("x", "52"); pct.setAttribute("y", "49"); pct.setAttribute("text-anchor", "middle");
        pct.setAttribute("dominant-baseline", "middle"); pct.setAttribute("fill", "#22d3ee");
        pct.style.fontSize = "18px"; pct.style.fontWeight = "700"; pct.textContent = "0%";
        const sub = document.createElementNS(NS, "text");
        sub.setAttribute("x", "52"); sub.setAttribute("y", "70"); sub.setAttribute("text-anchor", "middle");
        sub.setAttribute("fill", "#8b949e"); sub.style.fontSize = "10px"; sub.textContent = "step 0/0";
        svg.appendChild(pct); svg.appendChild(sub);
        header.appendChild(svg);

        const sum = document.createElement("div");
        sum.style.cssText = "display:flex;flex-direction:column;gap:4px;font-size:12px;";
        sum.innerHTML =
          '<div>ETA <b id="m-eta" style="color:#8ff08a">—</b></div>' +
          '<div>rate <b id="m-rate" style="color:#8ff08a">—</b></div>' +
          '<div>loss <b id="m-loss" style="color:#22d3ee">—</b></div>' +
          '<div>grads <b id="m-grads" style="color:#8ff08a">—</b></div>';
        header.appendChild(sum);
        root.appendChild(header);

        // ---- metric tiles grid ----
        const grid = document.createElement("div");
        grid.style.cssText = "display:grid;grid-template-columns:repeat(2,1fr);gap:6px;margin-top:10px;";
        function tile(label, id, color, unit) {
          const d = document.createElement("div");
          d.style.cssText = "background:#161b22;border:1px solid #21262d;border-radius:6px;padding:6px 8px;";
          d.innerHTML = `<div style="color:#8b949e;font-size:10px">${label}</div>` +
            `<div style="font-size:15px;font-weight:700;color:${color}"><span id="${id}">—</span> ${unit || ""}</div>`;
          grid.appendChild(d);
          return d;
        }
        tile("Video loss", "m-vloss", "#22d3ee");
        tile("Audio loss", "m-aloss", "#f0a35e");
        tile("s/step", "m-ss", "#e6edf3");
        tile("step/s", "m-sps", "#e6edf3");
        // VRAM tiles with a mini bar inside
        function vramTile(label, id, cap) {
          const d = document.createElement("div");
          d.style.cssText = "background:#161b22;border:1px solid #21262d;border-radius:6px;padding:6px 8px;";
          d.innerHTML = `<div style="color:#8b949e;font-size:10px">${label}</div>` +
            `<div style="font-size:15px;font-weight:700;color:#8ff08a"><span id="${id}">—</span> GB</div>` +
            `<div style="height:5px;background:#0b0f14;border-radius:3px;margin-top:3px;overflow:hidden">` +
            `<div id="${id}-bar" style="height:100%;width:0%;background:#22d3ee;transition:width .3s ease"></div></div>`;
          d.__cap = cap;
          grid.appendChild(d);
          return d;
        }
        vramTile("VRAM gpu0", "m-vr0", 12);
        vramTile("VRAM gpu1", "m-vr1", 12);
        vramTile("VRAM total", "m-vrt", 24);
        root.appendChild(grid);

        // ---- collapsible history chart ----
        const histBtn = document.createElement("button");
        histBtn.textContent = "History ▸";
        histBtn.style.cssText =
          "margin-top:8px;cursor:pointer;background:#161b22;color:#8ff08a;border:1px solid #21262d;" +
          "border-radius:6px;padding:3px 10px;font-size:11px;font-weight:bold;";
        const histWrap = document.createElement("div");
        histWrap.style.display = "none";
        const canvas = document.createElement("canvas");
        canvas.style.cssText = "width:100%;height:190px;background:#0b0f14;border:1px solid #21262d;border-radius:6px;display:block;margin-top:6px;";
        histWrap.appendChild(canvas);
        const hctx = canvas.getContext("2d");
        root.appendChild(histBtn); root.appendChild(histWrap);

        const widget = node.addDOMWidget("ltx25-metrics", "ltx25-metrics", root, {
          getValue: () => "", setValue: () => {}, serializeValue: () => "",
        });
        widget.serialize = false;
        widget.computeLayoutSize = () => ({ minHeight: 360, maxHeight: 420, minWidth: 380, maxWidth: 1e6 });

        let telPath = null, lastPct = -1;
        const set = (id, val) => { const el = root.querySelector("#" + id); if (el) el.textContent = val; };

        const drawHist = (losses, aLosses) => {
          const w = canvas.width, h = canvas.height;
          hctx.clearRect(0, 0, w, h);
          const n = getWindow();
          const data = losses.slice(-n), aData = aLosses.slice(-n);
          if (data.length < 2) { hctx.fillStyle = "#8b949e"; hctx.font = "11px Inter"; hctx.fillText("loss history…", 6, 14); return; }
          const min = Math.min(...data), max = Math.max(...data);
          const span = (max - min) || 1;
          const yFor = (v) => 6 + (1 - (v - min) / span) * (h - 20);
          hctx.strokeStyle = "#22d3ee"; hctx.lineWidth = 2; hctx.beginPath();
          data.forEach((v, i) => { const x = (i / (data.length - 1)) * w; const y = yFor(v); i === 0 ? hctx.moveTo(x, y) : hctx.lineTo(x, y); });
          hctx.stroke();
          hctx.strokeStyle = "#f0a35e"; hctx.lineWidth = 1.5; hctx.beginPath();
          let pen = false;
          aData.forEach((v, i) => {
            if (v == null) { pen = false; return; }
            const x = (i / (aData.length - 1)) * w; const y = yFor(v);
            if (!pen) { hctx.moveTo(x, y); pen = true; } else hctx.lineTo(x, y);
          });
          hctx.stroke();
          hctx.fillStyle = "#0b0f14"; hctx.fillRect(0, h - 16, w, 16);
          hctx.fillStyle = "#8b949e"; hctx.font = "11px Inter";
          hctx.fillText(`loss min ${min.toFixed(3)} max ${max.toFixed(3)} (last ${data.length})  cyan=total orange=audio`, 6, h - 4);
        };

        histBtn.addEventListener("click", () => {
          const open = histWrap.style.display === "none";
          histWrap.style.display = open ? "block" : "none";
          histBtn.textContent = open ? "History ▾" : "History ▸";
          if (open) setTimeout(() => drawHist(lastLosses, lastALosses), 50);
        });

        const resize = () => {
          const dpr = window.devicePixelRatio || 1;
          const rect = canvas.getBoundingClientRect();
          if (rect.width > 0) { canvas.width = rect.width * dpr; canvas.height = rect.height * dpr; hctx.setTransform(dpr, 0, 0, dpr, 0, 0); }
          if (histWrap.style.display !== "none") drawHist(lastLosses, lastALosses);
        };
        setTimeout(resize, 120);
        window.addEventListener("resize", resize);
        // Re-render at full resolution whenever the widget is resized (so enlarging
        // the node makes the chart actually bigger AND crisp, not stretched/blurry).
        if (typeof ResizeObserver !== "undefined") {
          new ResizeObserver(() => resize()).observe(histWrap);
        }

        let lastLosses = [], lastALosses = [];

        const onTelemetry = (ev) => { const d = ev && ev.detail; if (d && d.telemetry_path) telPath = d.telemetry_path; };
        api.addEventListener("ltx25:telemetry", onTelemetry);

        const timer = setInterval(async () => {
          try {
            if (!telPath) telPath = grabTelPath(node);
            if (!telPath) telPath = await deriveTelPath(node);
            if (!telPath) return;
            const evs = await fetchTelemetry(telPath);
            let total = 0, step = 0, loss = NaN, vl = NaN, al = NaN, eta = NaN, sps = NaN, ss = NaN, totalVRAM = 0;
            let grads = true, done = false;
            const losses = [], aLosses = [], vrams = [];
            for (const e of evs) {
              if (e.event === "start") total = e.steps || 0;
              else if (e.event === "done") { done = true; total = e.total_steps || total; }
              else if (e.event === "step") {
                if (e.total_steps) total = e.total_steps;
                if (e.step >= step) step = e.step;
                if (Number.isFinite(e.loss)) { loss = e.loss; losses.push(e.loss); }
                if (Number.isFinite(e.loss_video)) vl = e.loss_video;
                aLosses.push(Number.isFinite(e.loss_audio) ? e.loss_audio : null);
                if (Number.isFinite(e.loss_audio)) al = e.loss_audio;
                if (Number.isFinite(e.eta_s)) eta = e.eta_s;
                if (Number.isFinite(e.steps_per_sec)) sps = e.steps_per_sec;
                if (Number.isFinite(e.step_time)) ss = e.step_time;
                if (e.grads_finite === false) grads = false;
                if (e.peak_vram) vrams.push(e.peak_vram);
                if (Number.isFinite(e.peak_vram_gb)) totalVRAM = e.peak_vram_gb;
              }
            }
            if (total <= 0) total = await fetchRunTotal(telPath);
            if (done) step = total;
            const p = total > 0 ? (step / total) * 100 : 0;
            pct.textContent = Math.round(p) + "%";
            ring.setAttribute("stroke-dashoffset", C * (1 - p / 100));
            sub.textContent = `step ${step}/${total}`;
            set("m-eta", fmtEta(eta));
            set("m-rate", Number.isFinite(sps) ? sps.toFixed(2) : "—");
            set("m-loss", Number.isFinite(loss) ? loss.toFixed(4) : "—");
            set("m-grads", grads ? "ok" : "FAIL");
            set("m-vloss", Number.isFinite(vl) ? vl.toFixed(4) : "—");
            set("m-aloss", Number.isFinite(al) ? al.toFixed(4) : "—");
            set("m-ss", Number.isFinite(ss) ? ss.toFixed(2) : "—");
            set("m-sps", Number.isFinite(sps) ? sps.toFixed(2) : "—");
            const lastVR = vrams.length ? vrams[vrams.length - 1] : {};
            const g0 = lastVR.gpu0, g1 = lastVR.gpu1;
            const setBar = (id, val, cap) => {
              set(id, Number.isFinite(val) ? val.toFixed(1) : "—");
              const bar = root.querySelector("#" + id + "-bar");
              if (bar) bar.style.width = (Number.isFinite(val) ? Math.min(100, (val / cap) * 100) : 0) + "%";
            };
            setBar("m-vr0", g0, 12);
            setBar("m-vr1", g1, 12);
            set("m-vrt", totalVRAM > 0 ? totalVRAM.toFixed(1) : "—");
            const tbar = root.querySelector("#m-vrt-bar");
            if (tbar) tbar.style.width = (totalVRAM > 0 ? Math.min(100, (totalVRAM / 24) * 100) : 0) + "%";
            lastLosses = losses; lastALosses = aLosses;
            drawHist(losses, aLosses);
            if (Math.abs(p - lastPct) > 0.05) { lastPct = p; node.graph?.setDirtyCanvas(true, true); }
          } catch (e) { /* ignore */ }
        }, 1000);

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
          clearInterval(timer);
          api.removeEventListener("ltx25:telemetry", onTelemetry);
          onRemoved?.apply(this, arguments);
        };
      } catch (e) {
        console.error("[ltx25-metrics] setup error:", e);
      }
      return r;
    };
  },
};

app.registerExtension(EXT);
