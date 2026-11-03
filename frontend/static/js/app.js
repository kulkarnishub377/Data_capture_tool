/* Shared helpers + tab switching. Plain JS, no framework. */
const $ = (s, el = document) => el.querySelector(s);

function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) el.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k != null) el.append(k.nodeType ? k : document.createTextNode(k));
  return el;
}

let toastTimer;
function toast(msg, isErr = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = isErr ? "err" : "";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.textContent = ""), isErr ? 7000 : 3000);
}

async function api(path, opts = {}) {
  const init = { method: opts.method || "GET", headers: {} };
  if (opts.body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(opts.body);
  }
  let res;
  try { res = await fetch("/api" + path, init); }
  catch (e) { toast("Cannot reach the server", true); throw e; }
  const ct = res.headers.get("content-type") || "";
  const data = ct.includes("json") ? await res.json() : null;
  if (!res.ok) {
    const msg = (data && data.error) || `Request failed (${res.status})`;
    if (!opts.quiet) toast(msg, true);       // quiet: optional background lookups must not alarm the user
    throw new Error(msg);
  }
  return opts.raw ? res : data;
}

/* Theme: dark by default; the choice is remembered. The icon shows the theme you would switch TO. */
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  $("#theme-toggle i").className = t === "dark" ? "fa-solid fa-sun" : "fa-solid fa-moon";
  try { localStorage.setItem("theme", t); } catch (e) { /* storage unavailable */ }
}
$("#theme-toggle").addEventListener("click", () =>
  applyTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light"));
applyTheme(document.documentElement.dataset.theme || "dark");

const tabs = { live: {}, roi: {}, cameras: {}, gallery: {}, settings: {} };
function showTab(name) {
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach(s => s.classList.toggle("active", s.id === "tab-" + name));
  try { localStorage.setItem("tab", name); } catch (e) { /* storage unavailable */ }
  if (tabs[name].onShow) tabs[name].onShow();
}
$("#tabs").addEventListener("click", e => {
  const b = e.target.closest("button[data-tab]");
  if (b) showTab(b.dataset.tab);
});

/* "2026-11-02 10:30:15" (local time) -> "12s ago" */
function timeAgo(ts) {
  const t = new Date((ts || "").replace(" ", "T")).getTime();
  if (!t) return "";
  const s = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return ts.slice(0, 16);
}

/* Debounce: many quick changes -> one request */
function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

/* Shared full-size viewer with Next / Previous & Keyboard Navigation */
let lbItems = [];
let lbIndex = -1;
let lbOnDeleted = null;

function renderLightboxCurrent() {
  if (lbIndex < 0 || lbIndex >= lbItems.length) {
    closeLightbox();
    return;
  }
  const item = lbItems[lbIndex];
  const img = $("#lb-img");
  img.src = "/api/captures/" + item.path;
  img.alt = item.name || "Capture";

  const nameEl = $("#lb-name");
  if (nameEl) nameEl.textContent = item.ts ? `${item.ts} (${item.name})` : item.name;

  const countEl = $("#lb-counter");
  if (countEl) countEl.textContent = `${lbIndex + 1} / ${lbItems.length}`;

  const camEl = $("#lb-camera");
  if (camEl) camEl.textContent = item.camera || "Camera";

  const clsEl = $("#lb-class");
  if (clsEl) clsEl.textContent = (item.class || "Vehicle").replace(/_/g, " ");

  const kindEl = $("#lb-kind");
  if (kindEl) {
    kindEl.textContent = item.kind ? (item.kind === "crop" ? "Crop" : "Full frame") : "";
    kindEl.hidden = !item.kind;
  }

  const fullLink = $("#lb-full");
  if (fullLink) fullLink.href = "/api/captures/" + item.path;

  const prevBtn = $("#lb-prev");
  if (prevBtn) prevBtn.disabled = lbIndex <= 0;

  const nextBtn = $("#lb-next");
  if (nextBtn) nextBtn.disabled = lbIndex >= lbItems.length - 1;
}

function openLightbox(item, onDeleted, list = null, index = -1) {
  lbOnDeleted = onDeleted || null;
  if (Array.isArray(list) && list.length > 0) {
    lbItems = list.slice();
    lbIndex = index >= 0 ? index : lbItems.findIndex(x => x.path === item.path);
    if (lbIndex < 0) lbIndex = 0;
  } else {
    lbItems = [item];
    lbIndex = 0;
  }
  renderLightboxCurrent();
  $("#lightbox").hidden = false;
  document.body.style.overflow = "hidden";
}

function closeLightbox() {
  const lb = $("#lightbox");
  if (lb) lb.hidden = true;
  lbItems = [];
  lbIndex = -1;
  const img = $("#lb-img");
  if (img) img.removeAttribute("src");
  document.body.style.overflow = "";
}

function lbPrev() {
  if (lbIndex > 0) {
    lbIndex--;
    renderLightboxCurrent();
  }
}

function lbNext() {
  if (lbIndex < lbItems.length - 1) {
    lbIndex++;
    renderLightboxCurrent();
  }
}

async function lbDelete() {
  if (lbIndex < 0 || lbIndex >= lbItems.length) return;
  const item = lbItems[lbIndex];
  if (!confirm(`Delete capture "${item.name}"?`)) return;
  try {
    await api("/captures/" + item.path, { method: "DELETE" });
    toast("Image deleted");
    const cb = lbOnDeleted;
    lbItems.splice(lbIndex, 1);
    if (lbItems.length === 0) {
      closeLightbox();
    } else {
      if (lbIndex >= lbItems.length) lbIndex = lbItems.length - 1;
      renderLightboxCurrent();
    }
    if (cb) cb(item);
  } catch (e) { /* toast shown */ }
}

const lbCloseBtn = $("#lb-close");
if (lbCloseBtn) lbCloseBtn.addEventListener("click", closeLightbox);

const lbPrevBtn = $("#lb-prev");
if (lbPrevBtn) lbPrevBtn.addEventListener("click", lbPrev);

const lbNextBtn = $("#lb-next");
if (lbNextBtn) lbNextBtn.addEventListener("click", lbNext);

const lbDelBtn = $("#lb-del");
if (lbDelBtn) lbDelBtn.addEventListener("click", lbDelete);

const lbEl = $("#lightbox");
if (lbEl) {
  lbEl.addEventListener("click", e => {
    if (e.target.id === "lightbox" || e.target.classList.contains("lb-stage")) {
      closeLightbox();
    }
  });
}

document.addEventListener("keydown", e => {
  const lb = $("#lightbox");
  if (!lb || lb.hidden) return;
  if (e.key === "Escape") {
    e.preventDefault();
    closeLightbox();
  } else if (e.key === "ArrowLeft") {
    e.preventDefault();
    lbPrev();
  } else if (e.key === "ArrowRight") {
    e.preventDefault();
    lbNext();
  } else if (e.key === "Delete" || e.key === "Backspace") {
    if (!["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) {
      e.preventDefault();
      lbDelete();
    }
  }
});

let camerasCache = [];
async function loadCameras() {
  camerasCache = await api("/cameras");
  return camerasCache;
}

window.addEventListener("DOMContentLoaded", () => {
  let t = location.hash.slice(1) || "live";
  if (!location.hash) { try { t = localStorage.getItem("tab") || "live"; } catch (e) { /* ignore */ } }
  showTab(tabs[t] ? t : "live");
});
