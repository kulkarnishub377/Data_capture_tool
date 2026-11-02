/* Cameras tab: add, enable/disable, test, delete. */
(() => {
  const maskUrl = u => u.replace(/\/\/([^:@/]+):[^@]*@/, "//$1:****@");

  async function render() {
    const cams = await loadCameras();
    const list = $("#cam-list");
    list.replaceChildren();
    if (!cams.length) list.append(h("p", { class: "empty" }, "No cameras yet."));
    cams.forEach(c => {
      const en = h("input", { type: "checkbox" });
      en.checked = c.enabled !== false;
      en.addEventListener("change", async () => {
        try { await api(`/cameras/${c.id}`, { method: "PUT", body: { enabled: en.checked } }); }
        catch (e) { en.checked = !en.checked; }
      });
      const test = h("button", {}, "Test");
      test.addEventListener("click", async () => {
        test.disabled = true; toast("Testing connection...");
        try {
          const r = await api(`/cameras/${c.id}/test`, { method: "POST" });
          r.ok ? toast(`${c.name}: connected (${r.size[0]}x${r.size[1]})`) : toast(`${c.name}: ${r.error}`, true);
        } catch (e) { /* toast shown */ }
        test.disabled = false;
      });
      const del = h("button", { class: "danger" }, "Delete");
      del.addEventListener("click", async () => {
        if (!confirm(`Delete camera "${c.name}"? Captured images are kept.`)) return;
        try { await api(`/cameras/${c.id}`, { method: "DELETE" }); render(); } catch (e) { /* toast */ }
      });
      const roi = (c.roi || []).length >= 3 ? `ROI: ${c.roi.length} points` : "No ROI yet";
      list.append(h("div", { class: "row" },
        h("label", {}, en, " Enabled"),
        h("div", { class: "grow" }, h("strong", {}, c.name),
          h("small", {}, maskUrl(c.url)),
          h("small", {}, `${roi}${c.device ? " | device " + c.device : ""}`)),
        test, del));
    });
  }

  $("#cam-form").addEventListener("submit", async e => {
    e.preventDefault();
    try {
      await api("/cameras", { method: "POST", body: {
        name: $("#cam-name").value, url: $("#cam-url").value, device: $("#cam-device").value } });
      e.target.reset();
      toast("Camera added - draw its ROI in the ROI tab");
      render();
    } catch (err) { /* toast shown */ }
  });

  tabs.cameras.onShow = () => render().catch(() => {});
})();
