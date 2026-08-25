// O2noor System Monitor - top-bar widget (Crystools-style).
//
// Injects a live monitor into ComfyUI's top bar, positioned right before the
// settings group (the same spot Crystools / rgthree use), showing every GPU
// (util% / temp / mem) + RAM + CPU with colored, styled indicators.
// Always ON; not clickable; low-profile (lives inside the top bar, so it never
// covers the canvas).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

(function () {
  if (window.__o2noor_sysmon_topbar) return;
  window.__o2noor_sysmon_topbar = true;

  const wrap = document.createElement("div");
  wrap.id = "o2noor-sysmon-topbar";
  wrap.style.cssText =
    "display:flex;align-items:center;gap:10px;padding:0 10px;height:100%;" +
    "font-size:13px;font-family:Inter,system-ui,sans-serif;color:#8b949e;" +
    "white-space:nowrap;box-sizing:border-box;";

  const makeSeg = (color) => {
    const s = document.createElement("span");
    s.style.cssText =
      "display:inline-flex;align-items:center;gap:6px;padding:9px 14px;border-radius:6px;" +
      `background:rgba(255,255,255,.06);color:${color};font-weight:600;line-height:1;`;
    s.appendChild(dot(color));
    const txt = document.createElement("span");
    s.appendChild(txt);
    return { s, txt, dot: (c) => { s.querySelector("i").style.background = c; } };
  };
  const dot = (color) => {
    const i = document.createElement("i");
    i.style.cssText = `display:inline-block;width:6px;height:6px;border-radius:50%;background:${color};flex:0 0 auto;`;
    return i;
  };

  const gpu = {};
  let gpuCount = 0;
  const ram = makeSeg("#8ff08a");
  const cpu = makeSeg("#22d3ee");
  wrap.appendChild(ram.s);
  wrap.appendChild(cpu.s);

  const render = (data) => {
    if (!data || !data.ok) return;
    const gpus = data.gpus || [];
    if (gpus.length !== gpuCount) {
      gpuCount = gpus.length;
      for (const k in gpu) { const el = gpu[k]?.s; if (el && el.isConnected) el.remove(); }
      Object.keys(gpu).forEach((k) => delete gpu[k]);
      gpus.forEach((g, i) => {
        const seg = makeSeg("#22d3ee");
        seg.s.setAttribute("title", `${g.name}: ${g.mem_used_gb}/${g.mem_total_gb} GB`);
        wrap.insertBefore(seg.s, ram.s);
        gpu[i] = seg;
      });
    }
    gpus.forEach((g, i) => {
      const seg = gpu[i];
      if (!seg) return;
      seg.txt.textContent = `GPU${g.index} ${g.mem_used_gb}/${g.mem_total_gb}G ${g.util}% ${g.temp}°`;
      seg.dot(g.util > 90 ? "#f0a35e" : "#22d3ee");
      seg.s.setAttribute("title", `${g.name}: ${g.mem_used_gb}/${g.mem_total_gb} GB · ${g.util}% · ${g.temp}°C`);
    });
    if (data.ram) ram.txt.textContent = `RAM ${data.ram.used_gb}/${data.ram.total_gb}G ${data.ram.percent}%`;
    if (data.cpu) cpu.txt.textContent = `CPU ${data.cpu.percent}%`;
  };

  const tick = async () => {
    try {
      const resp = await api.fetchApi("/ltx25/system");
      render(await resp.json());
    } catch (e) { /* ignore */ }
  };

  // Insert into the top bar, before the settings group (same dock as rgthree/Crystools).
  const insert = () => {
    try {
      const menu = app.menu;
      const anchor = menu?.settingsGroup?.element;
      if (!anchor || !anchor.parentElement) return false;
      if (!wrap.isConnected) anchor.before(wrap);
      return true;
    } catch (e) { return false; }
  };

  const SETTING_ID = "Comfy.LTX25.SystemMonitor.Enabled";
  const isEnabled = () => {
    try { return app.ui?.settings?.getSettingValue(SETTING_ID) !== false; }
    catch (e) { return true; }
  };
  const ensureTimer = () => {
    if (!window.__o2noor_sysmon_started) {
      window.__o2noor_sysmon_started = true;
      tick();
      window.__o2noor_sysmon_timer = setInterval(tick, 2000);
      window.addEventListener("beforeunload", () => clearInterval(window.__o2noor_sysmon_timer));
    }
  };
  const stopTimer = () => {
    if (window.__o2noor_sysmon_timer) { clearInterval(window.__o2noor_sysmon_timer); window.__o2noor_sysmon_timer = null; }
    window.__o2noor_sysmon_started = false;
  };

  const showOrHide = () => {
    if (!isEnabled()) {
      if (wrap.isConnected) wrap.remove();
      stopTimer();
      return;
    }
    ensureTimer();
    if (!insert()) {
      const tries = window.__o2noor_sysmon_retries || 0;
      if (tries < 40) {
        window.__o2noor_sysmon_retries = tries + 1;
        setTimeout(showOrHide, 500);
      }
    }
  };

  const start = () => {
    showOrHide();
  };

  app.registerExtension({
    name: "O2noor.SystemMonitor.Topbar",
    setup() { start(); },
    settings: [
      {
        id: SETTING_ID,
        name: "O2noor System Monitor (top bar)",
        type: "boolean",
        defaultValue: true,
        category: ["O2noor", "System Monitor"],
        onChange: () => showOrHide(),
      },
    ],
  });
  start();
})();
