// LTX-2.5 System Monitor - live machine resource dashboard.
//
// One DOM widget on the O2noorLTX25Int4SystemMonitor node. Polls the /ltx25/system
// endpoint every second and renders:
//   - every GPU: name, memory used/total with a bar, utilization %, temperature,
//   - system RAM used/total with a bar,
//   - CPU usage % + core count.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXT = {
  name: "O2noor.LTX25.SystemMonitor",

  beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "O2noorLTX25Int4SystemMonitor") return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      try {
        if ((this.widgets || []).some((w) => w.name === "ltx25-system")) return r;

        const root = document.createElement("div");
        root.style.cssText =
          "width:100%;box-sizing:border-box;background:#0d1117;color:#e6edf3;" +
          "font-family:Inter,system-ui,sans-serif;padding:10px;border-radius:8px;";

        const title = document.createElement("div");
        title.style.cssText =
          "font-size:12px;font-weight:700;color:#22d3ee;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;";
        title.innerHTML =
          '<span>System Monitor</span><span id="sys-ts" style="color:#8b949e;font-weight:400;font-size:10px"></span>';
        root.appendChild(title);

        const bar = (valPct, color) => {
          const b = document.createElement("div");
          b.style.cssText = "height:6px;background:#0b0f14;border-radius:3px;margin-top:3px;overflow:hidden;";
          const f = document.createElement("div");
          f.style.cssText = `height:100%;width:0%;background:${color};transition:width .4s ease;`;
          b.appendChild(f);
          return { b, f };
        };
        const setPct = (f, pct) => { f.style.width = Math.max(0, Math.min(100, pct || 0)) + "%"; };

        // ---- GPU section ----
        const gpuWrap = document.createElement("div");
        gpuWrap.style.cssText = "display:flex;flex-direction:column;gap:8px;";
        root.appendChild(gpuWrap);

        // ---- RAM + CPU section ----
        const sysRow = document.createElement("div");
        sysRow.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px;";
        const makeSys = (label, idVal, idBar, color) => {
          const d = document.createElement("div");
          d.style.cssText = "background:#161b22;border:1px solid #21262d;border-radius:6px;padding:6px 8px;";
          d.innerHTML = `<div style="color:#8b949e;font-size:10px">${label}</div>` +
            `<div style="font-size:14px;font-weight:700;color:${color}"><span id="${idVal}">-</span></div>`;
          const { b, f } = bar(0, color);
          b.id = idBar;
          d.appendChild(b);
          return d;
        };
        const ramTile = makeSys("RAM", "sys-ram", "sys-ram-bar", "#8ff08a");
        const cpuTile = makeSys("CPU", "sys-cpu", "sys-cpu-bar", "#22d3ee");
        sysRow.appendChild(ramTile); sysRow.appendChild(cpuTile);
        root.appendChild(sysRow);

        const widget = this.addDOMWidget("ltx25-system", "ltx25-system", root, {
          getValue: () => "", setValue: () => {}, serializeValue: () => "",
        });
        widget.serialize = false;
        widget.computeLayoutSize = () => ({ minHeight: 200, maxHeight: 260, minWidth: 380, maxWidth: 1e6 });

        const set = (id, val) => { const el = root.querySelector("#" + id); if (el) el.textContent = val; };
        const gpuBarFor = {};

        const render = (data) => {
          if (!data || !data.ok) return;
          set("sys-ts", new Date().toLocaleTimeString());
          // GPUs
          const gpus = data.gpus || [];
          if (gpus.length !== gpuBarFor._count) {
            gpuWrap.innerHTML = "";
            gpuBarFor._count = gpus.length;
            gpus.forEach((g, i) => {
              const d = document.createElement("div");
              d.style.cssText = "background:#161b22;border:1px solid #21262d;border-radius:6px;padding:6px 8px;";
              d.innerHTML =
                `<div style="display:flex;justify-content:space-between;font-size:11px;color:#8b949e">` +
                `<span><b style="color:#e6edf3">GPU ${g.index}</b> ${g.name}</span>` +
                `<span><span id="g-${i}-mem" style="color:#8ff08a;font-weight:700">-</span>` +
                ` <span id="g-${i}-util" style="color:#22d3ee;font-weight:700">-</span>` +
                ` <span id="g-${i}-temp" style="color:#f0a35e;font-weight:700">-</span></span></div>`;
              const { b, f } = bar(0, "#22d3ee");
              d.appendChild(b);
              gpuWrap.appendChild(d);
              gpuBarFor[i] = { mem: d.querySelector("#g-" + i + "-mem"),
                              util: d.querySelector("#g-" + i + "-util"),
                              temp: d.querySelector("#g-" + i + "-temp"), f };
            });
          }
          gpus.forEach((g, i) => {
            const el = gpuBarFor[i];
            if (!el) return;
            el.mem.textContent = `${g.mem_used_gb}/${g.mem_total_gb} GB`;
            el.util.textContent = `${g.util}%`;
            el.temp.textContent = `${g.temp} C`;
            setPct(el.f, g.mem_total_gb > 0 ? (g.mem_used_gb / g.mem_total_gb) * 100 : 0);
          });
          // RAM / CPU
          if (data.ram) {
            set("sys-ram", `${data.ram.used_gb}/${data.ram.total_gb} GB (${data.ram.percent}%)`);
            const rf = root.querySelector("#sys-ram-bar > div");
            if (rf) setPct(rf, data.ram.percent);
          }
          if (data.cpu) {
            set("sys-cpu", `${data.cpu.percent}% (${data.cpu.cores_physical} phys / ${data.cpu.cores_logical} log)`);
            const cf = root.querySelector("#sys-cpu-bar > div");
            if (cf) setPct(cf, data.cpu.percent);
          }
          this.graph?.setDirtyCanvas(true, true);
        };

        const tick = async () => {
          try {
            const resp = await api.fetchApi("/ltx25/system");
            render(await resp.json());
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
        console.error("[ltx25-system] setup error:", e);
      }
      return r;
    };
  },
};

app.registerExtension(EXT);
