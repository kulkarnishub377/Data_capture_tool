/* Live tab: settings summary, camera grid with live view, recent captures panel.
   (All settings controls are in the Settings tab - see settings.js.) */
(() => {
  const cards = {};              // camera id -> card state
  let pollTimer = null;

  // ---------------------------------------------------------------- settings summary
  const SAVE_TEXT = { frame: "Full frame", crop: "Crop", both: "Frame + crop" };

  async function loadSummary() {
    const s = await api("/settings");
    const items = [
      ["Model", s.MODEL], ["Device", s.DEVICE], ["Confidence", s.CONF_THRESH],
      ["Classes", s.CLASSES.length ? `${s.CLASSES.length} selected` : "default"],
      ["Save", SAVE_TEXT[s.SAVE_MODE] || s.SAVE_MODE], ["When", s.WHEN === "first" ? "First in ROI" : "Best frame"],
      ["Metadata", s.SAVE_METADATA ? "Keep" : "Off"],
    ];
    $("#sum-items").replaceChildren(...items.map(([k, v]) => h("span", { class: "sum" }, h("b", {}, k), " ", String(v))));
  }
  $("#sum-open").addEventListener("click", () => showTab("settings"));

  // ---------------------------------------------------------------- camera cards
  function makeCard(cam) {
    const img = h("img", { alt: cam.name });
    const view = h("div", { class: "view" }, "Stopped");
    const chip = h("span", { class: "chip" }, "stopped");
    const count = h("span", { class: "count" }, "");
    const btn = h("button", { class: "primary" }, h("i", { class: "fa-solid fa-play" }), " Start");
    const c = { cam, img, view, chip, count, btn, running: false };
    btn.addEventListener("click", () => toggle(c));
    // if the live stream connection drops while the camera should be running, reconnect it
    img.addEventListener("error", () => {
      if (c.running) setTimeout(() => { if (c.running) img.src = `/api/stream/${cam.id}?t=${Date.now()}`; }, 1500);
    });
    const fs = h("button", { class: "fs", type: "button", title: "Full screen", "aria-label": `Full screen: ${cam.name}` },
      h("i", { class: "fa-solid fa-expand" }));
    fs.addEventListener("click", () => {
      if (document.fullscreenElement) document.exitFullscreen();
      else if (view.requestFullscreen) view.requestFullscreen();
    });
    c.el = h("div", { class: "cam" }, view,
      h("div", { class: "meta" }, h("span", { class: "name" }, cam.name), chip, count, fs, btn));
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
    const state = st ? st.state : "stopped";
    const running = state !== "stopped" && state !== "error";
    c.chip.textContent = state + (st && st.message && state !== "connected" ? ` - ${st.message}` : "");
    c.chip.className = "chip " + state;
    c.count.textContent = st ? `saved ${st.saved} | skipped ${st.skipped} | ${st.device}` : "";
    c.btn.replaceChildren(h("i", { class: running ? "fa-solid fa-stop" : "fa-solid fa-play" }), running ? " Stop" : " Start");
    c.btn.className = running ? "danger" : "primary";
    if (running && !c.running) {
      c.running = true;
      c.img.src = `/api/stream/${c.cam.id}?t=${Date.now()}`;
      c.view.replaceChildren(c.img);
    } else if (!running && c.running) {
      c.running = false;
      c.img.removeAttribute("src");
      c.view.replaceChildren(state === "error" ? "Stopped (see status)" : "Stopped");
    }
    c.running = running;
  }

  async function poll() {
    try {
      const st = await api("/capture/status");
      for (const id in cards) render(cards[id], st[id]);
    } catch (e) { /* offline: keep last state */ }
    pollRecent();
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
    $("#cam-count").textContent = cams.length ? `(${cams.length})` : "";
    await poll();
  }

  // ---------------------------------------------------------------- recent captures
  const recentEls = new Map();    // path -> element
  let recentFirst = true;

  async function pollRecent() {
    let r;
    try { r = await api("/recent?limit=10"); } catch (e) { return; }
    const list = $("#recent-list");
    const keep = new Set(r.items.map(i => i.path));
    for (const [path, el] of recentEls) if (!keep.has(path)) { el.remove(); recentEls.delete(path); }
    const ordered = r.items.map(it => {
      let el = recentEls.get(it.path);
      if (!el) {
        const img = h("img", { src: `/api/thumb/${it.path}?w=360`, loading: "lazy", alt: `${it.class} - ${it.camera}` });
        el = h("div", { class: "rc" + (recentFirst ? "" : " new"), title: it.name, tabindex: "0", role: "button" },
          img, h("span", { class: "tag" }, it.class),
          h("div", { class: "cap" }, h("span", {}, it.camera), h("span", { class: "ago" }, "")));
        const open = () => openLightbox(it, pollRecent);
        el.addEventListener("click", open);
        el.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
        recentEls.set(it.path, el);
      }
      el.querySelector(".ago").textContent = timeAgo(it.ts);
      return el;
    });
    list.replaceChildren(...ordered);
    recentFirst = false;
    $("#recent-empty").hidden = ordered.length > 0;
    $("#recent-count").textContent = ordered.length ? `latest ${ordered.length}` : "";
  }

  // ---------------------------------------------------------------- toolbar: start/stop all + layout
  function setCols(v) {
    const grid = $("#live-grid");
    document.querySelectorAll("#seg-cols button").forEach(b => b.setAttribute("aria-checked", String(b.dataset.value === v)));
    grid.classList.toggle("fixed", v !== "auto");
    if (v !== "auto") grid.style.setProperty("--cols", v);
    try { localStorage.setItem("liveCols", v); } catch (e) { /* storage unavailable */ }
  }
  $("#seg-cols").addEventListener("click", e => {
    const b = e.target.closest("button[data-value]");
    if (b) setCols(b.dataset.value);
  });
  try { setCols(localStorage.getItem("liveCols") || "auto"); } catch (e) { setCols("auto"); }

  async function bulk(path, button) {
    button.disabled = true;
    try { return await api(path, { method: "POST" }); }
    catch (e) { return null; }
    finally { button.disabled = false; poll(); }
  }
  $("#start-all").addEventListener("click", async () => {
    const r = await bulk("/capture/start-all", $("#start-all"));
    if (!r) return;
    const problems = Object.values(r.results).filter(x => x.error).map(x => `${x.name}: ${x.error}`);
    const parts = [];
    if (r.started) parts.push(`Started ${r.started}`);
    if (r.already_running) parts.push(`${r.already_running} already running`);
    if (r.skipped) parts.push(`${r.skipped} disabled`);
    if (problems.length) toast(`${parts.join(", ") || "Nothing started"} - could not start: ${problems.join("; ")}`, true);
    else toast(parts.join(", ") || "No cameras to start");
  });
  $("#stop-all").addEventListener("click", async () => {
    const r = await bulk("/capture/stop-all", $("#stop-all"));
    if (r) toast(r.stopped ? `Stopped ${r.stopped} camera${r.stopped === 1 ? "" : "s"}` : "No cameras were running");
  });

  tabs.live.onShow = async () => {
    try { await loadSummary(); await buildGrid(); } catch (e) { /* toast shown */ }
    clearInterval(pollTimer);
    pollTimer = setInterval(() => {
      if (!document.hidden && $("#tab-live").classList.contains("active")) poll();
    }, 2000);
  };
})();
