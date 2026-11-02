/* Live tab: settings bar + camera grid with live MJPEG view and Start/Stop. */
(() => {
  const cards = {};       // camera id -> {el, img, chip, count, btn, running}
  let pollTimer = null;

  const SETTING_CONTROLS = { "#sel-mode": "SAVE_MODE", "#sel-meta": "SAVE_METADATA", "#sel-when": "WHEN" };

  async function saveSetting(patch) {
    try { await api("/settings", { method: "PUT", body: patch }); toast("Saved - applies on next Start"); }
    catch (e) { /* toast shown by api() */ }
  }

  async function loadClasses(model) {
    const r = await api("/model/classes?model=" + encodeURIComponent(model));
    const bar = $("#class-bar");
    bar.replaceChildren(h("span", { class: "hint" }, "Classes:"));
    const chosen = new Set(r.selected);
    r.names.forEach(n => {
      const cb = h("input", { type: "checkbox", value: n });
      cb.checked = chosen.has(n);
      cb.addEventListener("change", () => {
        const sel = [...bar.querySelectorAll("input:checked")].map(i => i.value);
        if (!sel.length) { cb.checked = true; toast("Select at least one class", true); return; }
        saveSetting({ CLASSES: sel });
      });
      bar.append(h("label", {}, cb, " ", n));
    });
  }

  async function loadControls() {
    const [models, s, dev] = await Promise.all([api("/models"), api("/settings"), api("/devices")]);
    const sm = $("#sel-model");
    sm.replaceChildren(...models.map(m => h("option", { value: m }, m)));
    if (!models.includes(s.MODEL)) sm.prepend(h("option", { value: s.MODEL }, s.MODEL + " (missing)"));
    sm.value = s.MODEL;

    const opts = ["auto", "cpu", ...dev.gpus.map(g => g.id)];
    if (dev.gpus.length > 1) opts.push(dev.gpus.map(g => g.id).join(","));
    if (!opts.includes(s.DEVICE)) opts.push(s.DEVICE);
    const label = v => {
      const g = dev.gpus.find(x => x.id === v);
      return g ? `${v} - ${g.name}` : v;
    };
    $("#sel-device").replaceChildren(...opts.map(o => h("option", { value: o }, label(o))));
    $("#sel-device").value = s.DEVICE;
    if (dev.warnings.length) toast(dev.warnings.join("; "), true);

    $("#sel-mode").value = s.SAVE_MODE;
    $("#sel-meta").value = String(s.SAVE_METADATA);
    $("#sel-when").value = s.WHEN;
    if (models.includes(s.MODEL)) await loadClasses(s.MODEL);
  }

  $("#sel-model").addEventListener("change", async e => {
    try {
      await api("/settings", { method: "PUT", body: { MODEL: e.target.value, CLASSES: [] } });
      await loadClasses(e.target.value);
      toast("Model saved - applies on next Start");
    } catch (err) { loadControls(); }
  });
  $("#sel-device").addEventListener("change", e => saveSetting({ DEVICE: e.target.value }));
  for (const [sel, key] of Object.entries(SETTING_CONTROLS)) {
    $(sel).addEventListener("change", e => {
      const v = e.target.value;
      saveSetting({ [key]: key === "SAVE_METADATA" ? v === "true" : v });
    });
  }

  function makeCard(cam) {
    const img = h("img", { alt: cam.name });
    const view = h("div", { class: "view" }, "Stopped");
    const chip = h("span", { class: "chip" }, "stopped");
    const count = h("span", { class: "count" }, "");
    const btn = h("button", { class: "primary" }, "Start");
    const c = { cam, img, view, chip, count, btn, running: false };
    btn.addEventListener("click", () => toggle(c));
    c.el = h("div", { class: "cam" }, view,
      h("div", { class: "meta" }, h("span", { class: "name" }, cam.name), chip, count, btn));
    return c;
  }

  async function toggle(c) {
    c.btn.disabled = true;
    try {
      if (c.running) {
        await api(`/capture/${c.cam.id}/stop`, { method: "POST" });
      } else {
        const r = await api(`/capture/${c.cam.id}/start`, { method: "POST" });
        if (r.warnings && r.warnings.length) toast(r.warnings.join("; "), true);
        else toast(`Started on ${r.device}`);
      }
    } catch (e) { /* toast shown */ }
    c.btn.disabled = false;
    poll();
  }

  function render(c, st) {
    const running = !!st && st.state !== "stopped";
    c.chip.textContent = st ? st.state + (st.message && st.state !== "connected" ? ` - ${st.message}` : "") : "stopped";
    c.chip.className = "chip " + (st ? st.state : "");
    c.count.textContent = st ? `saved ${st.saved} | skipped ${st.skipped} | ${st.device}` : "";
    c.btn.textContent = running ? "Stop" : "Start";
    c.btn.className = running ? "danger" : "primary";
    if (running && !c.running) {
      c.img.src = `/api/stream/${c.cam.id}?t=${Date.now()}`;
      c.view.replaceChildren(c.img);
    } else if (!running && c.running) {
      c.img.removeAttribute("src");
      c.view.replaceChildren("Stopped");
    }
    c.running = running;
  }

  async function poll() {
    try {
      const st = await api("/capture/status");
      for (const id in cards) render(cards[id], st[id]);
    } catch (e) { /* offline: keep last state */ }
  }

  async function buildGrid() {
    const cams = await loadCameras();
    const grid = $("#live-grid");
    grid.replaceChildren();
    for (const k in cards) delete cards[k];
    cams.filter(c => c.enabled !== false).forEach(cam => {
      cards[cam.id] = makeCard(cam);
      grid.append(cards[cam.id].el);
    });
    $("#live-empty").hidden = cams.length > 0;
    await poll();
  }

  tabs.live.onShow = async () => {
    try { await loadControls(); await buildGrid(); } catch (e) { /* toast shown */ }
    clearInterval(pollTimer);
    pollTimer = setInterval(() => { if (document.getElementById("tab-live").classList.contains("active")) poll(); }, 2000);
  };
})();
