// O2noor LTX 2.5 Dataset Timeline - live animated pipeline dashboard.
//
// One DOM widget on the O2noorLTX25Int4DatasetTimeline node. Polls the pack's
// /ltx25/dataset_timeline?path=... every second and renders everything the Voice
// Dataset node is doing inside (so it never looks frozen):
//   - current stage with a moving spinner + elapsed timer,
//   - ffmpeg clip-cutting progress (done/total, ~s/clip, elapsed),
//   - audio extract / caption encode / precompute / VAE encode bars,
//   - per-model load times (seconds) with a running total,
//   - a live, auto-scrolling status event log.
// The dataset root is auto-followed from the connected Voice Dataset node when
// dataset_path is left empty, so it watches the exact directory being encoded.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXT = {
  name: "O2noor.LTX25.DatasetTimeline",

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "O2noorLTX25Int4DatasetTimeline") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const that = this;
      let timer = null;

      const pollWidget = (this.widgets || []).find((w) => w.name === "poll_seconds");

      const root = document.createElement("div");
      root.style.cssText =
        "width:100%;box-sizing:border-box;background:#0d1117;color:#e6edf3;" +
        "font-family:Inter,system-ui,sans-serif;padding:12px;border-radius:10px;" +
        "overflow:hidden;";
      root.appendChild(titleBar());

      const canvas = document.createElement("div");
      canvas.style.cssText = "display:flex;flex-direction:column;gap:10px;margin-top:8px;";
      root.appendChild(canvas);

      const widget = this.addDOMWidget("ltx25-dataset-timeline", "ltx25-dataset-timeline", root, {
        getValue: () => "", setValue: () => {}, serializeValue: () => "",
      });
      widget.serialize = false;
      widget.computeLayoutSize = () => ({ minHeight: 240, maxHeight: 560, minWidth: 400, maxWidth: 1e6 });

      function titleBar() {
        const t = document.createElement("div");
        t.style.cssText =
          "font-size:13px;font-weight:800;color:#22d3ee;display:flex;align-items:center;justify-content:space-between;";
        t.innerHTML =
          '<span><span class="dtl-spinner" style="display:inline-block;width:12px;height:12px;margin-right:6px;vertical-align:middle"></span>' +
          "Dataset Timeline</span>" +
          '<span style="color:#8b949e;font-weight:400;font-size:10px" class="dtl-clock"></span>';
        return t;
      }

      const spinner = () => {
        // Style the spinner as a static ring (no animation). Animation is driven
        // only by render() while a stage is actually running, so it never spins
        // idly before/without a wired dataset.
        const s = root.querySelector(".dtl-spinner");
        if (s) {
          s.style.border = "2px solid rgba(34,211,238,.2)";
          s.style.borderTopColor = "#22d3ee";
          s.style.borderRadius = "50%";
          s.style.animation = "none";
        }
      };
      const css = document.createElement("style");
      css.textContent =
        "@keyframes dtlspin{to{transform:rotate(360deg)}}" +
        ".dtl-prog{height:7px;background:#0b0f14;border-radius:4px;overflow:hidden;position:relative;}" +
        ".dtl-prog>i{position:absolute;inset:0;background:linear-gradient(90deg,#22d3ee,#3b82f6);" +
        "width:0%;transition:width .5s ease;border-radius:4px;}" +
        ".dtl-prog.running>i::after{content:\"\";position:absolute;inset:0;" +
        "background:linear-gradient(90deg,transparent,rgba(255,255,255,.35),transparent);" +
        "animation:dtlshimmer 1.2s linear infinite;}" +
        "@keyframes dtlshimmer{from{transform:translateX(-100%)}to{transform:translateX(100%)}}" +
        ".dtl-row{background:#161b22;border:1px solid #21262d;border-radius:7px;padding:7px 9px;}" +
        ".dtl-lbl{font-size:10px;color:#8b949e;display:flex;justify-content:space-between;}" +
        ".dtl-val{font-size:13px;font-weight:700;color:#e6edf3;}";
      document.head.appendChild(css);

      const pad = (n) => String(n).padStart(2, "0");
      const hm = (s) => {
        s = Math.max(0, Math.floor(s));
        const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = s % 60;
        return (h ? h + ":" + pad(m) : m) + ":" + pad(ss);
      };

      const render = (data, path) => {
        // clock
        const c = root.querySelector(".dtl-clock");
        if (c) c.textContent = new Date().toLocaleTimeString();

        // ---- Empty state: no wired dataset / no data -> explicit message, NO spinner ----
        const wired = isWired();
        const pathEmpty = !path || !path.trim();
        if (!wired || pathEmpty || !data || !data.ok) {
          const spinEl = root.querySelector(".dtl-spinner");
          if (spinEl) spinEl.style.animation = "none";   // stop the endless circle
          canvas.innerHTML = "";
          const msg = document.createElement("div");
          msg.style.cssText =
            "background:#161b22;border:1px dashed #30363d;border-radius:7px;" +
            "padding:14px;text-align:center;color:#8b949e;font-size:12px;line-height:1.6;";
          msg.innerHTML = wired
            ? 'Waiting for the dataset output to produce data…<br>' +
              '<span style="font-size:10px;color:#6e7681">Run the Voice Dataset node upstream to populate the timeline.</span>'
            : 'Wire the <b style="color:#e6edf3">dataset</b> output here.<br>' +
              '<span style="font-size:10px;color:#6e7681">Connect O2noor LTX 2.5 Voice Dataset → dataset, then run it. No path is guessed.</span>';
          canvas.appendChild(msg);
          ctx.setDirtyCanvas(true, true);
          return;
        }

        // ---- Stage card ----
        const stage = data.events && data.events.length ? data.events[data.events.length - 1].stage : "idle";
        const stageLabels = {
          start: "Initializing dataset",
          cut: "Cutting clips (ffmpeg)", cut_done: "Clip cutting done",
          audio_extract: "Extracting voice tracks",
          encode_captions: "Encoding captions (Gemma)",
          encode_captions_done: "Captions encoded",
          precompute_audio_embeds: "Precomputing audio embeds",
          precompute_audio_embeds_done: "Audio embeds done",
          vae_encode: "Encoding VAE latents",
          vae_encode_done: "VAE encode done",
          done: "Complete",
        };
        const live = (["cut", "audio_extract", "encode_captions", "precompute_audio_embeds", "vae_encode"].indexOf(stage) >= 0);
        // Drive the spinner only while a stage is actually running.
        const spinEl = root.querySelector(".dtl-spinner");
        if (spinEl) spinEl.style.animation = live ? "dtlspin .8s linear infinite" : "none";

        canvas.innerHTML = "";
        const stageCard = card("Current stage");
        stageCard.appendChild(
          rowVal('<span class="dtl-live" style="color:' + (live ? "#f0a35e" : "#8ff08a") + '">' +
            (live ? "● " : "") + (stageLabels[stage] || stage) + "</span>", "") +
          rowVal('<span style="color:#8b949e;font-weight:400;font-size:10px">' +
            (data.events && data.events.length ? "last event " + (data.events[data.events.length - 1].ts || "") : "waiting for events") + "</span>", ""));
        canvas.appendChild(stageCard);

        // ---- Clip cutting (scenes) ----
        const scenes = data.scenes || {};
        const target = (data.dataset && data.dataset.samples) || 0;
        const cutCard = card("ffmpeg clip cutting  ·  ~" + (scenes.sample_rate ? scenes.sample_rate.toFixed(2) + " clip/s" : "—"));
        const cutDone = scenes.count || 0;
        cutCard.appendChild(rowVal(cutDone + " / " + (target || "?") + " clips", cutDone + " clips cut"));
        cutCard.appendChild(progress(cutDone, target));
        cutCard.appendChild(rowVal("elapsed " + hm((scenes.newest_t ? Date.now() / 1000 - scenes.newest_t : 0)), ""));
        canvas.appendChild(cutCard);

        // ---- Stage-specific progress bars ----
        const evs = data.events || [];
        const vaeEv = [...evs].reverse().find((e) => e.stage === "vae_encode" && e.sub === "video");
        const audEv = [...evs].reverse().find((e) => e.stage === "vae_encode" && e.sub === "audio");
        if (vaeEv) {
          const c = card("Video VAE encode (face)");
          c.appendChild(rowVal((vaeEv.done || 0) + " / " + (vaeEv.total || 0), "video latents"));
          c.appendChild(progress(vaeEv.done || 0, vaeEv.total || 0, true));
          canvas.appendChild(c);
        }
        if (audEv) {
          const c = card("Audio VAE encode (voice)");
          c.appendChild(rowVal((audEv.done || 0) + " / " + (audEv.total || 0), "audio latents"));
          c.appendChild(progress(audEv.done || 0, audEv.total || 0, true));
          canvas.appendChild(c);
        }

        // ---- Model load times ----
        const loads = data.loads || [];
        if (loads.length) {
          const c = card("Model load times  ·  total " + hm(loads.reduce((a, l) => a + (l.load_s || 0), 0)));
          loads.forEach((l) => {
            c.appendChild(rowVal(l.component + "", (l.load_s || 0) + " s"));
            c.appendChild(progress(null, null, false, (l.load_s || 0) / Math.max(1, Math.max(...loads.map((x) => x.load_s || 0)))));
          });
          canvas.appendChild(c);
        }

        // ---- Live event log tail ----
        if (evs.length) {
          const c = card("Event log");
          const log = document.createElement("div");
          log.style.cssText = "max-height:110px;overflow-y:auto;font-family:ui-monospace,monospace;" +
            "font-size:10px;color:#8b949e;line-height:1.5;";
          [...evs].slice(-14).forEach((e) => {
            const line = document.createElement("div");
            line.style.whiteSpace = "nowrap";
            line.textContent = (e.ts || "") + "  " + (e.stage || "") +
              (e.done != null ? "  " + e.done + "/" + (e.total || "?") : "") +
              (e.note ? "  " + e.note : "");
            log.appendChild(line);
          });
          c.appendChild(log);
          canvas.appendChild(c);
        }

        ctx.setDirtyCanvas(true, true);
      };

      function card(label) {
        const d = document.createElement("div");
        d.style.cssText = "background:#161b22;border:1px solid #21262d;border-radius:7px;padding:7px 9px;";
        d.innerHTML = '<div style="font-size:10px;color:#22d3ee;font-weight:700;margin-bottom:5px">' + label + "</div>";
        return d;
      }
      function rowVal(left, right) {
        const d = document.createElement("div");
        d.style.cssText = "display:flex;justify-content:space-between;align-items:baseline;font-size:12px;";
        d.innerHTML = '<span style="color:#8b949e">' + left + '</span><span style="color:#e6edf3;font-weight:700">' + right + "</span>";
        return d;
      }
      function progress(done, total, running, frac) {
        const b = document.createElement("div");
        b.className = "dtl-prog" + (running ? " running" : "");
        const f = document.createElement("i");
        let pct = 0;
        if (frac != null) pct = Math.max(0, Math.min(100, frac * 100));
        else if (total) pct = Math.max(0, Math.min(100, (done / total) * 100));
        f.style.width = pct + "%";
        b.appendChild(f);
        return b;
      }

      // Live status path delivered by the Voice Dataset node at encode start via
      // ltx25:dataset_status (mirrors how Train pushes ltx25:telemetry). When set,
      // it takes priority over wiring so the widget polls the live encode.
      let liveStatusPath = null;
      const onDatasetStatus = (event) => {
        const d = event && event.detail;
        if (d && d.status_path) liveStatusPath = d.status_path;
      };
      api.addEventListener("ltx25:dataset_status", onDatasetStatus);

      // ---- Resolve the monitored dataset root ----
      // Priority: 1) live path broadcast by Voice Dataset at encode start,
      // 2) the connected dataset node's output dict (dataset_root, via getInputData),
      // 3) the graph-linked source node's output_dir widget. No hardcoded paths.
      function resolvePath() {
        if (liveStatusPath && liveStatusPath.trim()) return liveStatusPath.trim();
        try {
          const d = that.getInputData ? that.getInputData(0) : null;
          if (d && d.dataset_root) return d.dataset_root;
        } catch (e) { /* ignore */ }
        try {
          const input = (that.inputs || []).find((i) => i.name === "dataset");
          if (input && input.link != null) {
            const link = that.graph && that.graph.links[input.link];
            if (link) {
              const src = that.graph.getNodeById(link.origin_id);
              if (src && src.widgets && src.widgets.length) {
                const w = src.widgets.find((x) => x.name === "output_dir" || x.name === "dataset_root");
                if (w && w.value) return w.value;
              }
            }
          }
        } catch (e) { /* ignore */ }
        return "";
      }

      const isWired = () => {
        if (liveStatusPath && liveStatusPath.trim()) return true;
        try {
          const d = that.getInputData ? that.getInputData(0) : null;
          if (d && d.dataset_root) return true;
          const input = (that.inputs || []).find((i) => i.name === "dataset");
          return !!(input && input.link != null);
        } catch (e) { return false; }
      };

      // ---- Poll loop ----
      async function tick() {
        const path = resolvePath();
        try {
          const url = "/ltx25/dataset_timeline?path=" + encodeURIComponent(path);
          const resp = await api.fetchApi(url);
          render(await resp.json(), path);
        } catch (e) { /* ignore */ }
      }

      spinner();  // static ring styling only (idle = not rotating)
      tick();
      const poll = () => Math.max(0.5, Number(pollWidget ? pollWidget.value : 1) || 1) * 1000;
      timer = setInterval(tick, poll());

      const onRemoved = nodeType.prototype.onRemoved;
      nodeType.prototype.onRemoved = function () {
        clearInterval(timer);
        try { api.removeEventListener("ltx25:dataset_status", onDatasetStatus); } catch (e) {}
        onRemoved?.apply(this, arguments);
      };
      return r;
    };
  },
};

app.registerExtension(EXT);
