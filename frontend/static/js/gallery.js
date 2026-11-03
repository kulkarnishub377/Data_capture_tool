/* Gallery tab: browse captured images with multi-category checkbox filtering,
   batch selection, search, pagination, and keyboard-enabled lightbox. */
(() => {
  let page = 1;
  let total = 0;
  let perPage = 60;
  let sort = "newest";
  let kind = "";
  let searchTimer = null;

  const knownCategories = new Set();
  const selectedCategories = new Set();
  const selectedPaths = new Set();
  let currentItems = [];
  let filtersPopulated = false;

  // ---------------------------------------------------------------- Category Checkboxes
  function renderCategoryChips() {
    const list = $("#g-category-list");
    if (!list) return;
    list.replaceChildren();

    const sorted = Array.from(knownCategories).sort();
    sorted.forEach(cls => {
      const isChecked = selectedCategories.has(cls);
      const input = h("input", {
        type: "checkbox",
        value: cls,
        checked: isChecked,
        onchange: () => {
          if (input.checked) {
            selectedCategories.add(cls);
          } else {
            selectedCategories.delete(cls);
          }
          label.classList.toggle("active", input.checked);
          updateCategorySummary();
          filtersChanged();
        }
      });

      const label = h("label", { class: "g-cat-chip" + (isChecked ? " active" : "") },
        input,
        h("span", { class: "g-cat-name" }, cls.replace(/_/g, " "))
      );
      list.append(label);
    });

    updateCategorySummary();
  }

  function updateCategorySummary() {
    const countBadge = $("#g-cat-count");
    if (!countBadge) return;
    if (knownCategories.size === 0) {
      countBadge.textContent = "No categories yet";
      return;
    }
    if (selectedCategories.size === knownCategories.size) {
      countBadge.textContent = `All ${knownCategories.size} selected`;
    } else {
      countBadge.textContent = `${selectedCategories.size} of ${knownCategories.size} selected`;
    }
  }

  function selectAllCategories() {
    knownCategories.forEach(c => selectedCategories.add(c));
    renderCategoryChips();
    filtersChanged();
  }

  function clearAllCategories() {
    selectedCategories.clear();
    renderCategoryChips();
    filtersChanged();
  }

  // ---------------------------------------------------------------- Batch Selection
  function updateBatchBar() {
    const bar = $("#g-batch-bar");
    if (!bar) return;
    const count = selectedPaths.size;
    bar.hidden = count === 0;
    const countEl = $("#g-batch-count");
    if (countEl) countEl.textContent = count;
    const delCountEl = $("#g-delete-count");
    if (delCountEl) delCountEl.textContent = count;

    // Synchronize card selection classes
    document.querySelectorAll(".g-card").forEach(card => {
      const p = card.dataset.path;
      const isSelected = selectedPaths.has(p);
      card.classList.toggle("selected", isSelected);
      const cb = card.querySelector(".g-card-cb");
      if (cb) cb.checked = isSelected;
    });
  }

  function selectPage() {
    currentItems.forEach(it => selectedPaths.add(it.path));
    updateBatchBar();
  }

  function selectNone() {
    selectedPaths.clear();
    updateBatchBar();
  }

  async function deleteSelected() {
    const count = selectedPaths.size;
    if (count === 0) return;
    if (!confirm(`Permanently delete ${count} selected capture${count === 1 ? "" : "s"}?`)) return;

    let res;
    try {
      res = await api("/captures/delete-batch", { method: "POST", body: { paths: Array.from(selectedPaths) } });
    } catch (e) {
      return;                                   // nothing was deleted; api() already showed why
    }
    const deleted = res.deleted || 0;
    const errors = res.errors || [];
    const plural = n => (n === 1 ? "" : "s");
    if (errors.length) {
      toast(`Deleted ${deleted} image${plural(deleted)}; ${errors.length} could not be deleted (${errors[0]})`, true);
    } else {
      toast(`Deleted ${deleted} image${plural(deleted)}`);
    }
    selectedPaths.clear();
    updateBatchBar();
    await load();
  }

  /* A filter changed: back to page 1, and drop the selection so "Delete selected" never removes images
     that are no longer visible. Paging and sorting keep the selection. */
  function filtersChanged() {
    page = 1;
    selectedPaths.clear();
    updateBatchBar();
    load();
  }

  // ---------------------------------------------------------------- Filters Initialization
  /* Camera choices = the camera folders that hold captures (as returned by the server), so images of a camera
     that was renamed or deleted can still be filtered. The current choice is kept even if it has no images. */
  function fillCameraSelect(folders) {
    const sc = $("#g-cam");
    if (!sc || !Array.isArray(folders)) return;
    const keep = sc.value;
    const names = new Set(folders);
    if (keep) names.add(keep);
    const sorted = Array.from(names).sort();
    const current = Array.from(sc.options).map(o => o.value).slice(1);
    if (current.length === sorted.length && current.every((v, i) => v === sorted[i])) return;
    sc.replaceChildren(h("option", { value: "" }, "All cameras"), ...sorted.map(n => h("option", { value: n }, n)));
    sc.value = keep;
  }

  async function fillFilters() {
    // Discover categories from model
    try {
      const modelRes = await api("/model/classes", { quiet: true });   // optional: captures on disk also give classes
      if (modelRes && Array.isArray(modelRes.names)) {
        modelRes.names.forEach(n => {
          if (!knownCategories.has(n)) {
            knownCategories.add(n);
            if (!filtersPopulated) selectedCategories.add(n);
          }
        });
      }
    } catch (e) { /* model classes unavailable */ }

    filtersPopulated = true;
    renderCategoryChips();
  }

  // ---------------------------------------------------------------- Load Captures
  function showEmpty(title, hint) {
    const t = $("#g-empty-title");
    if (t) t.textContent = title;
    const hn = $("#g-empty-hint");
    if (hn) hn.textContent = hint;
  }

  let loadSeq = 0;   // only the newest request may draw: fast typing / clicking must not show stale results

  async function load() {
    const seq = ++loadSeq;
    if (knownCategories.size > 0 && selectedCategories.size === 0) {
      total = 0;
      currentItems = [];
      showEmpty("No categories selected", "Tick at least one category, or press Select all.");
      renderGrid([]);
      renderPagination(0);
      return;
    }

    const q = new URLSearchParams({ page: String(page), per_page: String(perPage), sort });
    const camVal = $("#g-cam")?.value;
    if (camVal) q.set("camera", camVal);                    // a camera folder name, exactly as listed
    const dateVal = $("#g-date")?.value;
    if (dateVal) q.set("date", dateVal);
    if (kind) q.set("kind", kind);
    const searchVal = ($("#g-search")?.value || "").trim();
    if (searchVal) q.set("q", searchVal);                   // searched on the server, across all pages
    const classFiltered = knownCategories.size > 0 && selectedCategories.size < knownCategories.size;
    if (classFiltered) q.set("class", Array.from(selectedCategories).join(","));

    let r;
    try {
      r = await api("/captures?" + q);
    } catch (e) {
      return;
    }
    if (seq !== loadSeq) return;                            // a newer request has been sent meanwhile

    total = r.total || 0;
    currentItems = r.items || [];
    fillCameraSelect(r.cameras);

    // Categories that exist on disk but not in the current model are learnt (and selected)
    let added = false;
    (r.categories || []).forEach(c => {
      if (!knownCategories.has(c)) {
        knownCategories.add(c);
        selectedCategories.add(c);
        added = true;
      }
    });
    if (added) {
      renderCategoryChips();
      if (classFiltered) return load();                     // the request did not include the new ones yet
    }

    // e.g. the last images of the last page were deleted: go to the page that now exists
    const pages = Math.max(1, Math.ceil(total / perPage));
    if (currentItems.length === 0 && total > 0 && page > pages) {
      page = pages;
      return load();
    }

    showEmpty("No captures found", "Try adjusting your camera, category, date or search filters.");
    renderGrid(currentItems);
    renderPagination(currentItems.length);
  }

  // ---------------------------------------------------------------- Render Cards Grid
  function renderGrid(items) {
    const grid = $("#g-grid");
    const emptyEl = $("#g-empty");
    if (!grid) return;
    grid.replaceChildren();

    if (items.length === 0) {
      if (emptyEl) emptyEl.hidden = false;
      return;
    }
    if (emptyEl) emptyEl.hidden = true;

    items.forEach((it, idx) => {
      const isSelected = selectedPaths.has(it.path);

      // Checkbox for batch select
      const cb = h("input", {
        type: "checkbox",
        class: "g-card-cb",
        checked: isSelected,
        "aria-label": `Select ${it.name}`,
        onchange: e => {
          e.stopPropagation();
          if (cb.checked) {
            selectedPaths.add(it.path);
          } else {
            selectedPaths.delete(it.path);
          }
          card.classList.toggle("selected", cb.checked);
          updateBatchBar();
        }
      });

      const cbWrap = h("label", {
        class: "g-card-select-wrap",
        title: "Select image",
        onclick: e => e.stopPropagation()
      }, cb);

      // Class badge
      const clsBadge = h("span", { class: "badge g-card-cls" }, (it.class || "vehicle").replace(/_/g, " "));

      // Kind badge (crop vs frame)
      const kindBadge = h("span", { class: "badge sm g-card-kind" }, it.kind === "crop" ? "Crop" : "Frame");

      // Top overlays container
      const topOverlay = h("div", { class: "g-card-overlay-top" }, cbWrap, h("div", { class: "g-card-badges" }, clsBadge, kindBadge));

      // Thumbnail Image
      const img = h("img", {
        src: "/api/thumb/" + it.path + "?w=320",
        loading: "lazy",
        alt: it.name || "Capture"
      });
      const thumbWrap = h("div", { class: "g-thumb-wrap" }, img);

      // Footer info & quick single delete
      const delBtn = h("button", {
        type: "button",
        class: "btn danger sm g-card-del",
        title: "Delete this image",
        onclick: async e => {
          e.stopPropagation();
          if (!confirm(`Delete image "${it.name}"?`)) return;
          try {
            await api("/captures/" + it.path, { method: "DELETE" });
            toast("Image deleted");
            selectedPaths.delete(it.path);
            updateBatchBar();
            load();
          } catch (err) { /* toast shown */ }
        }
      }, h("i", { class: "fa-solid fa-trash" }));

      const infoRow = h("div", { class: "g-card-info" },
        h("div", { class: "g-card-meta-text" },
          h("span", { class: "g-card-cam" }, h("i", { class: "fa-solid fa-camera" }), " ", it.camera || ""),
          h("span", { class: "g-card-time", title: it.ts || it.name },
            h("i", { class: "fa-regular fa-clock" }), " ", timeAgo(it.ts) || it.date || "")
        ),
        delBtn
      );

      const card = h("div", {
        class: "g-card" + (isSelected ? " selected" : ""),
        "data-path": it.path,
        onclick: () => openLightbox(it, gone => { selectedPaths.delete(gone.path); updateBatchBar(); load(); }, items, idx)
      }, topOverlay, thumbWrap, infoRow);

      grid.append(card);
    });

    updateBatchBar();
  }

  // ---------------------------------------------------------------- Pagination
  function renderPagination(displayedCount) {
    const pages = Math.max(1, Math.ceil(total / perPage));
    const countText = $("#g-count-text");
    if (countText) {
      countText.textContent = `${total} total capture${total === 1 ? "" : "s"}`;
    }

    const summary = $("#g-page-summary");
    if (summary) {
      const start = total === 0 ? 0 : (page - 1) * perPage + 1;
      const end = Math.min(total, (page - 1) * perPage + displayedCount);
      summary.textContent = total > 0 ? `Showing ${start}-${end} of ${total}` : "0 captures";
    }

    const pagePill = $("#g-page");
    if (pagePill) pagePill.textContent = `Page ${page} / ${pages}`;

    const firstBtn = $("#g-first");
    if (firstBtn) firstBtn.disabled = page <= 1;

    const prevBtn = $("#g-prev");
    if (prevBtn) prevBtn.disabled = page <= 1;

    const nextBtn = $("#g-next");
    if (nextBtn) nextBtn.disabled = page >= pages;

    const lastBtn = $("#g-last");
    if (lastBtn) lastBtn.disabled = page >= pages;
  }

  // ---------------------------------------------------------------- Reset Filters
  function resetAllFilters() {
    const cam = $("#g-cam");
    if (cam) cam.value = "";

    const date = $("#g-date");
    if (date) date.value = "";

    const search = $("#g-search");
    if (search) search.value = "";

    kind = "";
    document.querySelectorAll("#g-seg-kind button").forEach(b => {
      b.setAttribute("aria-checked", String(b.dataset.value === ""));
    });

    const sortSel = $("#g-sort");
    if (sortSel) sortSel.value = "newest";
    sort = "newest";

    const perPageSel = $("#g-per-page");
    if (perPageSel) perPageSel.value = "60";
    perPage = 60;

    knownCategories.forEach(c => selectedCategories.add(c));
    renderCategoryChips();

    page = 1;
    selectedPaths.clear();
    updateBatchBar();
    load();
    toast("Filters reset");
  }

  // ---------------------------------------------------------------- Event Listeners
  // Category buttons
  $("#g-cat-all")?.addEventListener("click", selectAllCategories);
  $("#g-cat-none")?.addEventListener("click", clearAllCategories);

  // Batch action buttons
  $("#g-select-all-page")?.addEventListener("click", selectPage);
  $("#g-select-none")?.addEventListener("click", selectNone);
  $("#g-delete-selected")?.addEventListener("click", deleteSelected);

  // Filter inputs
  $("#g-cam")?.addEventListener("change", filtersChanged);
  $("#g-date")?.addEventListener("change", filtersChanged);

  $("#g-date-today")?.addEventListener("click", () => {
    const d = new Date();
    const iso = d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, "0") + "-" + String(d.getDate()).padStart(2, "0");
    const dateInput = $("#g-date");
    if (dateInput) dateInput.value = iso;
    filtersChanged();
  });

  $("#g-date-clear")?.addEventListener("click", () => {
    const dateInput = $("#g-date");
    if (dateInput) dateInput.value = "";
    filtersChanged();
  });

  // Image type segmented control
  $("#g-seg-kind")?.addEventListener("click", e => {
    const btn = e.target.closest("button[data-value]");
    if (!btn) return;
    kind = btn.dataset.value || "";
    document.querySelectorAll("#g-seg-kind button").forEach(b => {
      b.setAttribute("aria-checked", String(b.dataset.value === kind));
    });
    filtersChanged();
  });

  // Search input with debounce
  $("#g-search")?.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(filtersChanged, 300);
  });

  // Sort & Per page
  $("#g-sort")?.addEventListener("change", e => {
    sort = e.target.value || "newest";
    page = 1;
    load();
  });

  $("#g-per-page")?.addEventListener("change", e => {
    perPage = parseInt(e.target.value, 10) || 60;
    page = 1;
    load();
  });

  // Refresh & Reset
  $("#g-refresh")?.addEventListener("click", () => load());
  $("#g-reset-filters")?.addEventListener("click", resetAllFilters);
  $("#g-empty-reset")?.addEventListener("click", resetAllFilters);

  // Pagination buttons
  $("#g-first")?.addEventListener("click", () => { if (page > 1) { page = 1; load(); } });
  $("#g-prev")?.addEventListener("click", () => { if (page > 1) { page--; load(); } });
  $("#g-next")?.addEventListener("click", () => {
    const pages = Math.max(1, Math.ceil(total / perPage));
    if (page < pages) { page++; load(); }
  });
  $("#g-last")?.addEventListener("click", () => {
    const pages = Math.max(1, Math.ceil(total / perPage));
    if (page < pages) { page = pages; load(); }
  });

  // Tab activation
  tabs.gallery.onShow = async () => {
    try {
      await fillFilters();
      await load();
    } catch (e) {
      /* toast shown */
    }
  };
})();
