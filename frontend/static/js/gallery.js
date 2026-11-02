/* Gallery tab: browse captured images (filters, paging, enlarge, delete). */
(() => {
  let page = 1, total = 0, perPage = 60, current = null, filtersReady = false;

  async function fillFilters() {
    const cams = await loadCameras();
    const sc = $("#g-cam"), keep = sc.value;
    sc.replaceChildren(h("option", { value: "" }, "All"), ...cams.map(c => h("option", { value: c.name }, c.name)));
    sc.value = keep;
    if (!filtersReady) {
      try {
        const m = (await api("/model/classes")).names;
        $("#g-cls").replaceChildren(h("option", { value: "" }, "All"), ...m.map(n => h("option", { value: n }, n)));
      } catch (e) { /* model missing: leave class filter as All */ }
      filtersReady = true;
    }
  }

  async function load() {
    const q = new URLSearchParams({ page, per_page: perPage });
    if ($("#g-cam").value) q.set("camera", $("#g-cam").value.replace(/[^A-Za-z0-9._-]+/g, "_"));
    if ($("#g-date").value) q.set("date", $("#g-date").value);
    if ($("#g-cls").value) q.set("class", $("#g-cls").value);
    const r = await api("/captures?" + q);
    total = r.total;
    const grid = $("#g-grid");
    grid.replaceChildren();
    if (!r.items.length) grid.append(h("p", { class: "empty" }, "No captures match."));
    r.items.forEach(it => {
      const img = h("img", { src: "/api/captures/" + it.path, loading: "lazy", alt: it.name });
      const fig = h("figure", { title: it.name }, img, h("figcaption", {}, `${it.camera} | ${it.class} | ${it.kind}`));
      fig.addEventListener("click", () => open(it));
      grid.append(fig);
    });
    const pages = Math.max(1, Math.ceil(total / perPage));
    $("#g-count").textContent = `${total} image${total === 1 ? "" : "s"}`;
    $("#g-page").textContent = `Page ${page} / ${pages}`;
    $("#g-prev").disabled = page <= 1;
    $("#g-next").disabled = page >= pages;
  }

  function open(it) {
    current = it;
    $("#lb-img").src = "/api/captures/" + it.path;
    $("#lb-name").textContent = it.name;
    $("#lightbox").hidden = false;
  }
  const close = () => { $("#lightbox").hidden = true; current = null; };
  $("#lb-close").addEventListener("click", close);
  $("#lightbox").addEventListener("click", e => { if (e.target.id === "lightbox") close(); });
  document.addEventListener("keydown", e => { if (e.key === "Escape") close(); });
  $("#lb-del").addEventListener("click", async () => {
    if (!current || !confirm("Delete this image?")) return;
    try { await api("/captures/" + current.path, { method: "DELETE" }); close(); load(); } catch (e) { /* toast */ }
  });

  ["#g-cam", "#g-date", "#g-cls"].forEach(s => $(s).addEventListener("change", () => { page = 1; load(); }));
  $("#g-refresh").addEventListener("click", () => load());
  $("#g-prev").addEventListener("click", () => { page--; load(); });
  $("#g-next").addEventListener("click", () => { page++; load(); });

  tabs.gallery.onShow = async () => { try { await fillFilters(); await load(); } catch (e) { /* toast */ } };
})();
