/* Settings tab: model, device, detection, capture options and thresholds.

   Nothing is applied while you edit. Changes collect in a draft; the single "Save & restart" button sends
   them in one request and the server restarts any running camera once (saved images and counters are kept).
   "Discard" returns every control to the saved values. Rows and sections with unsaved edits are marked. */
(() => {
  let settings = {};                       // what the server has (saved)
  let draft = {};                          // key -> new value, only for controls that differ from `settings`
  let names = [], defaults = [], selected = new Set(), baseSelected = new Set();
  let saving = false;

  const same = (a, b) => (Array.isArray(a) || Array.isArray(b))
    ? JSON.stringify([...(a || [])].sort()) === JSON.stringify([...(b || [])].sort())
    : a === b;

  // ---------------------------------------------------------------- section navigation
  function showSection(name) {
    document.querySelectorAll(".set-sec").forEach(s => { s.hidden = s.id !== "sec-" + name; });
    document.querySelectorAll("#set-nav button").forEach(b => b.classList.toggle("active", b.dataset.sec === name));
    try { localStorage.setItem("setSec", name); } catch (e) { /* storage unavailable */ }
  }
  $("#set-nav").addEventListener("click", e => {
    const b = e.target.closest("button[data-sec]");
    if (b) showSection(b.dataset.sec);
  });
  try { showSection(localStorage.getItem("setSec") || "model"); } catch (e) { showSection("model"); }

  // ---------------------------------------------------------------- draft + save bar
  function setDraft(patch) {
    for (const [k, v] of Object.entries(patch)) {
      if (k !== "CLASSES" && same(v, settings[k])) delete draft[k];
      else draft[k] = v;
    }
    updateBar();
  }

  function setPill(kind, icon, text) {
    const pill = $("#set-status");
    pill.className = "pill " + kind;
    pill.querySelector("i").className = "fa-solid " + icon;
    $("#set-status-text").textContent = text;
  }

  function updateBar() {
    const keys = Object.keys(draft);
    const n = keys.length;
    if (saving) setPill("busy", "fa-spinner fa-spin", "Saving...");
    else if (n) setPill("dirty", "fa-circle-exclamation", `${n} unsaved change${n === 1 ? "" : "s"}`);
    else setPill("ok", "fa-circle-check", "All changes saved");
    $("#set-save").disabled = saving || n === 0;
    $("#set-discard").disabled = saving || n === 0;
    $("#tab-btn-settings").classList.toggle("dirty", n > 0);

    // mark the rows and sections that hold unsaved edits
    document.querySelectorAll(".srow").forEach(r => {
      r.classList.toggle("changed", (r.dataset.keys || "").split(",").some(k => k in draft));
    });
    document.querySelectorAll(".set-sec").forEach(sec => {
      const btn = document.querySelector(`#set-nav button[data-sec="${sec.id.slice(4)}"]`);
      if (btn) btn.classList.toggle("has-changes", !!sec.querySelector(".srow.changed"));
    });
  }

  async function runningCount() {
    try {
      const st = await api("/capture/status");
      return Object.values(st).filter(s => s.state !== "stopped" && s.state !== "error").length;
    } catch (e) { return 0; }
  }

  async function saveAll() {
    if (!Object.keys(draft).length || saving) return;
    saving = true; updateBar();
    try {
      const running = await runningCount();
      settings = await api("/settings", { method: "PUT", body: draft });
      draft = {};
      toast(running ? `Saved - restarting ${running} running camera${running === 1 ? "" : "s"} (saved images are kept)` : "Saved");
      await loadControls();
    } catch (e) { /* the reason is shown as a toast; the draft stays so you can fix it */ }
    saving = false;
    updateBar();
  }

  async function discard() {
    draft = {};
    await loadControls();
  }

  $("#set-save").addEventListener("click", saveAll);
  $("#set-discard").addEventListener("click", discard);
  window.addEventListener("beforeunload", e => {
    if (Object.keys(draft).length) { e.preventDefault(); e.returnValue = ""; }
  });

  // ---------------------------------------------------------------- class picker
  const MAX_CHIPS = 10;

  function classesChanged() {
    const modelChanged = "MODEL" in draft;       // a new model always resets the saved classes
    if (same([...selected], [...baseSelected])) {
      if (modelChanged) draft.CLASSES = []; else delete draft.CLASSES;
    } else draft.CLASSES = [...selected];
    updateBar();
  }

  function renderChips() {
    const list = [...selected];
    const chips = list.slice(0, MAX_CHIPS).map(n => {
      const x = h("button", { type: "button", title: `Remove ${n}`, "aria-label": `Remove ${n}` }, h("i", { class: "fa-solid fa-xmark" }));
      x.addEventListener("click", () => toggleClass(n, false));
      return h("span", { class: "chip-x" }, n, x);
    });
    if (list.length > MAX_CHIPS) chips.push(h("span", { class: "chip-more" }, `+${list.length - MAX_CHIPS} more`));
    $("#class-chips").replaceChildren(...chips);
    $("#class-btn-text").textContent = `Choose classes (${list.length} of ${names.length})`;
  }

  function renderClassList() {
    const q = $("#class-search").value.trim().toLowerCase();
    $("#class-list").replaceChildren(...names.filter(n => n.toLowerCase().includes(q)).map(n => {
      const cb = h("input", { type: "checkbox" });
      cb.checked = selected.has(n);
      cb.addEventListener("change", () => toggleClass(n, cb.checked));
      return h("label", {}, cb, n);
    }));
  }

  function toggleClass(n, on) {
    if (on) selected.add(n);
    else {
      if (selected.size <= 1) { toast("Keep at least one class selected", true); renderClassList(); return; }
      selected.delete(n);
    }
    renderChips(); renderClassList(); classesChanged();
  }

  /* Load the class list of `model`. useDefaults: show the model's default set (used when the model was changed). */
  async function loadClasses(model, useDefaults = false) {
    const r = await api("/model/classes?model=" + encodeURIComponent(model));
    names = r.names; defaults = r.default;
    selected = new Set(useDefaults ? r.default : r.selected);
    baseSelected = new Set(useDefaults ? r.default : r.selected);
    if (!useDefaults && r.warnings && r.warnings.length) toast(r.warnings.join("; "), true);
    const coco = names.includes("person") && names.includes("traffic light");
    const info = $("#model-info");
    info.hidden = false;
    info.textContent = `${coco ? "COCO" : "Custom"} - ${names.length} classes`;
    renderChips(); renderClassList();
  }

  function closePicker() { $("#class-pop").hidden = true; $("#class-btn").setAttribute("aria-expanded", "false"); }
  $("#class-btn").addEventListener("click", e => {
    e.stopPropagation();
    const open = $("#class-pop").hidden;
    $("#class-pop").hidden = !open;
    $("#class-btn").setAttribute("aria-expanded", String(open));
    if (open) { $("#class-search").value = ""; renderClassList(); $("#class-search").focus(); }
  });
  $("#class-pop").addEventListener("click", e => e.stopPropagation());
  document.addEventListener("click", closePicker);
  document.addEventListener("keydown", e => { if (e.key === "Escape") closePicker(); });
  $("#class-search").addEventListener("input", renderClassList);
  document.querySelectorAll("#class-pop [data-preset]").forEach(b => b.addEventListener("click", () => {
    selected = new Set(b.dataset.preset === "all" ? names : defaults);
    renderChips(); renderClassList(); classesChanged();
  }));

  // ---------------------------------------------------------------- controls
  const NUM = {   // element -> [setting key, parser]
    "#adv-minbox": ["MIN_BOX", v => parseInt(v, 10)],
    "#adv-overlap": ["MIN_ROI_OVERLAP", parseFloat],
    "#adv-dedup": ["DEDUP_SECONDS", parseFloat],
    "#adv-maxhold": ["MAX_HOLD_SECONDS", parseFloat],
    "#adv-jpeg": ["JPEG_QUALITY", v => parseInt(v, 10)],
    "#adv-imgsz": ["IMG_SIZE", v => parseInt(v, 10)],
  };

  function segSet(id, value) {
    document.querySelectorAll(`#${id} button`).forEach(b => b.setAttribute("aria-checked", String(b.dataset.value === value)));
  }
  document.querySelectorAll(".seg").forEach(seg => seg.addEventListener("click", e => {
    const b = e.target.closest("button[data-value]");
    if (!b) return;
    segSet(seg.id, b.dataset.value);
    setDraft({ [seg.dataset.key]: b.dataset.value });
  }));

  function metaText() { $("#meta-text").textContent = $("#sel-meta").checked ? "On" : "Off"; }
  $("#sel-meta").addEventListener("change", e => { metaText(); setDraft({ SAVE_METADATA: e.target.checked }); });

  /* Fill every control from the server (used on first load, after Save and on Discard). */
  async function loadControls() {
    const [models, s, dev, paths] = await Promise.all([api("/models"), api("/settings"), api("/devices"), api("/paths")]);
    settings = s;
    const info = $("#models-dir-info");
    info.className = "path-info" + (paths.models_dir_exists ? "" : " bad");
    info.textContent = paths.models_dir_exists
      ? `Models folder: ${paths.models_dir} (${paths.models_count} model${paths.models_count === 1 ? "" : "s"})`
      : `Models folder not found: ${paths.models_dir} - fix MODELS_DIR in config.ini and restart the service`;
    const sm = $("#sel-model");
    sm.replaceChildren(...models.map(m => h("option", { value: m }, m)));
    if (!models.includes(s.MODEL)) sm.prepend(h("option", { value: s.MODEL }, s.MODEL + " (missing)"));
    sm.value = s.MODEL;

    const opts = ["auto", "cpu", ...dev.gpus.map(g => g.id)];
    if (dev.gpus.length > 1) opts.push(dev.gpus.map(g => g.id).join(","));
    if (!opts.includes(s.DEVICE)) opts.push(s.DEVICE);
    const label = v => { const g = dev.gpus.find(x => x.id === v); return g ? `${v} - ${g.name}` : v; };
    $("#sel-device").replaceChildren(...opts.map(o => h("option", { value: o }, label(o))));
    $("#sel-device").value = s.DEVICE;
    $("#dev-info").textContent = dev.gpus.length
      ? `${dev.gpus.length} GPU${dev.gpus.length === 1 ? "" : "s"} detected: ${dev.gpus.map(g => g.name).join(", ")}. With several GPUs, cameras are spread across them.`
      : "No GPU detected - running on the CPU.";
    if (dev.warnings.length) toast(dev.warnings.join("; "), true);

    segSet("seg-mode", s.SAVE_MODE);
    segSet("seg-when", s.WHEN);
    $("#sel-meta").checked = !!s.SAVE_METADATA; metaText();
    $("#conf-range").value = s.CONF_THRESH;
    $("#conf-num").value = s.CONF_THRESH;
    for (const [sel, [key]] of Object.entries(NUM)) $(sel).value = s[key];
    setModelStatus("");
    if (models.includes(s.MODEL)) {
      try { await loadClasses(s.MODEL); }
      catch (e) { setModelStatus(e.message, true); }          // keep the rest of the page usable, show WHY it failed
    }
    updateBar();
  }

  function setModelStatus(text, bad = false) {
    const el = $("#model-status");
    el.hidden = !text;
    el.textContent = text;
    el.className = "path-info" + (bad ? " bad" : " ok");
  }

  $("#env-check").addEventListener("click", async () => {
    const btn = $("#env-check"), box = $("#env-result");
    btn.disabled = true; box.hidden = false; box.replaceChildren(h("span", { class: "hint" }, "Checking..."));
    try {
      const f = await api("/environment");
      const arm = /aarch64|arm64/i.test(f.machine);
      const row = (k, v, bad) => h("div", { class: "env-row" + (bad ? " bad" : "") }, h("b", {}, k), " ", v);
      const t = f.torch || {}, rt = f.tensorrt || {}, cv = f.opencv || {};
      const rows = [
        row("Python", `${f.version} (${f.machine}) ${f.in_venv ? "in a venv" + (f.system_site_packages ? " with system packages" : " WITHOUT system packages") : "(system)"}`, f.in_venv && !f.system_site_packages && arm),
        row("PyTorch", t.error ? "not available" : `${t.version}, CUDA ${t.cuda ? "yes" + (t.gpu ? " (" + t.gpu + ")" : "") : "NO"}`, !!t.error || !t.cuda),
        row("TensorRT", rt.error ? "not available" : rt.version, !!rt.error && arm),
        row("OpenCV", cv.error ? "not available" : `${cv.version}, GStreamer ${cv.gstreamer ? "yes" : "no"}, FFmpeg ${cv.ffmpeg ? "yes" : "no"}`, !!cv.error || !cv.ffmpeg),
        row("Ultralytics", f.ultralytics || "not installed", !f.ultralytics),
        row("Tracker (lap)", f.lap ? "yes" : "missing", !f.lap),
      ];
      const warns = (f.warnings || []).map(w => h("div", { class: "env-warn" }, h("i", { class: "fa-solid fa-triangle-exclamation" }), " ", w));
      box.replaceChildren(...rows, ...(warns.length ? warns : [h("div", { class: "env-ok" }, h("i", { class: "fa-solid fa-circle-check" }), " Environment looks fine.")]));
    } catch (e) { box.replaceChildren(h("span", { class: "path-info bad" }, e.message)); }
    btn.disabled = false;
  });

  $("#model-test").addEventListener("click", async () => {
    const btn = $("#model-test"), model = $("#sel-model").value;
    btn.disabled = true;
    setModelStatus("Testing: loading the model and running a test frame on the selected device...");
    try {
      const r = await api(`/model/test?model=${encodeURIComponent(model)}`, { method: "POST" });
      setModelStatus(r.ok ? `OK - ${r.message}` : `FAILED - ${r.error}`, !r.ok);
    } catch (e) { setModelStatus(e.message, true); }
    btn.disabled = false;
  });

  $("#sel-model").addEventListener("change", async e => {
    const model = e.target.value;
    if (model === settings.MODEL) { delete draft.MODEL; delete draft.CLASSES; }
    else { draft.MODEL = model; draft.CLASSES = []; }       // new model -> its default classes
    updateBar();
    try { await loadClasses(model, model !== settings.MODEL); } catch (err) { /* toast shown */ }
  });
  $("#sel-device").addEventListener("change", e => setDraft({ DEVICE: e.target.value }));
  $("#conf-range").addEventListener("input", e => { $("#conf-num").value = e.target.value; setDraft({ CONF_THRESH: parseFloat(e.target.value) }); });
  $("#conf-num").addEventListener("change", e => {
    const v = parseFloat(e.target.value);
    if (!(v >= 0.01 && v <= 1)) { toast("Confidence must be between 0.01 and 1", true); e.target.value = draft.CONF_THRESH ?? settings.CONF_THRESH; return; }
    $("#conf-range").value = v;
    setDraft({ CONF_THRESH: v });
  });
  for (const [sel, [key, parse]] of Object.entries(NUM)) {
    $(sel).addEventListener("change", e => {
      const v = parse(e.target.value);
      if (Number.isNaN(v)) { toast("Enter a number", true); e.target.value = draft[key] ?? settings[key]; return; }
      setDraft({ [key]: v });
    });
  }

  // keep an unsaved draft when you switch tabs and come back
  tabs.settings.onShow = () => { if (!Object.keys(draft).length) loadControls().catch(() => {}); else updateBar(); };
})();
