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
    toast(msg, true);
    throw new Error(msg);
  }
  return opts.raw ? res : data;
}

const tabs = { live: {}, roi: {}, cameras: {}, gallery: {} };
function showTab(name) {
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach(s => s.classList.toggle("active", s.id === "tab-" + name));
  try { localStorage.setItem("tab", name); } catch (e) { /* storage unavailable */ }
  if (tabs[name].onShow) tabs[name].onShow();
}
$("#tabs").addEventListener("click", e => { if (e.target.dataset.tab) showTab(e.target.dataset.tab); });

let camerasCache = [];
async function loadCameras() {
  camerasCache = await api("/cameras");
  return camerasCache;
}

window.addEventListener("DOMContentLoaded", () => {
  let t = "live";
  try { t = localStorage.getItem("tab") || "live"; } catch (e) { /* ignore */ }
  showTab(tabs[t] ? t : "live");
});
