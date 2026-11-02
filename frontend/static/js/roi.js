/* ROI tab: several ROI polygons per camera. Click to add points to the active ROI, drag points to move,
   "+ New ROI" to start another area. A vehicle is captured when it is inside ANY ROI.
   Points are stored in true camera-frame pixels. */
(() => {
  const canvas = $("#roi-canvas");
  const ctx = canvas.getContext("2d");
  const COLORS = ["#facc15", "#22d3ee", "#f472b6", "#a3e635", "#fb923c", "#c084fc"];
  const R = 8;                // point hit radius (display px)
  let img = null;             // snapshot image
  let frameSize = null;       // [w, h] of the real camera frame
  let rois = [];              // [[ [x, y], ... ], ...]
  let active = -1;            // index of the ROI being edited
  let drag = null;            // {roi, point}

  const color = i => COLORS[i % COLORS.length];
  const scale = () => (frameSize ? canvas.width / frameSize[0] : 1);

  function problem() {
    if (!frameSize) return "Load a snapshot first.";
    if (!rois.length) return "Click on the picture to add the first ROI point.";
    const bad = rois.findIndex(r => r.length < 3);
    return bad >= 0 ? `ROI ${bad + 1} needs at least 3 points.` : "";
  }

  function renderChips() {
    const box = $("#roi-chips");
    box.replaceChildren(...rois.map((r, i) => {
      const b = h("button", { class: i === active ? "active" : "", title: "Select this ROI" }, `ROI ${i + 1} (${r.length} pts)`);
      b.style.borderLeftColor = color(i);
      b.addEventListener("click", () => { active = i; refresh(); });
      return b;
    }));
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (img) ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    else {
      ctx.fillStyle = "#9aa3b2"; ctx.font = "16px system-ui";
      ctx.fillText("Choose a camera and click 'Load snapshot'", 20, 30);
    }
    const s = scale();
    rois.forEach((pts, i) => {
      if (!pts.length) return;
      const c = color(i), on = i === active;
      ctx.beginPath();
      pts.forEach(([x, y], k) => (k ? ctx.lineTo(x * s, y * s) : ctx.moveTo(x * s, y * s)));
      if (pts.length >= 3) { ctx.closePath(); ctx.globalAlpha = on ? 0.28 : 0.15; ctx.fillStyle = c; ctx.fill(); ctx.globalAlpha = 1; }
      ctx.strokeStyle = c; ctx.lineWidth = on ? 3 : 2; ctx.stroke();
      pts.forEach(([x, y], k) => {
        ctx.beginPath(); ctx.arc(x * s, y * s, on ? R - 1 : R - 3, 0, 7);
        ctx.fillStyle = k === 0 ? "#22c55e" : c; ctx.fill();
      });
      ctx.fillStyle = c; ctx.font = "bold 14px system-ui";
      ctx.fillText(`ROI ${i + 1}`, pts[0][0] * s + 10, pts[0][1] * s - 10);
    });
  }

  function refresh() {
    draw(); renderChips();
    const p = problem();
    $("#roi-save").disabled = !!p;
    if (p) $("#roi-hint").textContent = p;
    else $("#roi-hint").textContent = `${rois.length} ROI${rois.length === 1 ? "" : "s"} ready. Vehicles inside any of them are captured. Press "Save ROIs".`;
  }

  function pos(e) {
    const r = canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * (canvas.width / r.width), (e.clientY - r.top) * (canvas.height / r.height)];
  }
  function hit([cx, cy]) {
    const s = scale();
    for (let i = rois.length - 1; i >= 0; i--) {
      const k = rois[i].findIndex(([x, y]) => Math.hypot(x * s - cx, y * s - cy) <= R + 4);
      if (k >= 0) return { roi: i, point: k };
    }
    return null;
  }

  canvas.addEventListener("pointerdown", e => {
    if (!frameSize) { toast("Load a snapshot first", true); return; }
    const p = pos(e), hp = hit(p);
    if (hp) { drag = hp; active = hp.roi; canvas.setPointerCapture(e.pointerId); refresh(); return; }
    if (active < 0 || !rois[active]) { rois.push([]); active = rois.length - 1; }
    const s = scale();
    rois[active].push([Math.round(p[0] / s), Math.round(p[1] / s)]);
    refresh();
  });
  canvas.addEventListener("pointermove", e => {
    if (!drag) return;
    const [cx, cy] = pos(e), s = scale();
    rois[drag.roi][drag.point] = [Math.max(0, Math.min(frameSize[0], Math.round(cx / s))),
                                  Math.max(0, Math.min(frameSize[1], Math.round(cy / s)))];
    draw();
  });
  canvas.addEventListener("pointerup", () => { drag = null; refresh(); });

  async function loadSnapshot() {
    const id = $("#roi-cam").value;
    if (!id) return toast("Add a camera first", true);
    $("#roi-load").disabled = true;
    toast("Grabbing a frame (up to ~15 s)...");
    try {
      const res = await api(`/cameras/${id}/snapshot?t=${Date.now()}`, { raw: true });
      const m = /^(\d+)x(\d+)$/.exec(res.headers.get("X-Frame-Size") || "");
      const blob = await res.blob();
      const im = new Image();
      im.onload = () => {
        img = im;
        frameSize = m ? [+m[1], +m[2]] : [im.naturalWidth, im.naturalHeight];
        canvas.height = Math.round(canvas.width * frameSize[1] / frameSize[0]);
        refresh(); toast("Snapshot loaded");
      };
      im.src = URL.createObjectURL(blob);
    } catch (e) { /* the server's reason is already shown as a toast */ }
    $("#roi-load").disabled = false;
  }

  $("#roi-load").addEventListener("click", loadSnapshot);
  $("#roi-new").addEventListener("click", () => {
    if (rois.length && rois[rois.length - 1].length === 0) { active = rois.length - 1; return refresh(); }
    rois.push([]); active = rois.length - 1; refresh();
  });
  $("#roi-undo").addEventListener("click", () => {
    if (active >= 0 && rois[active]) { rois[active].pop(); if (!rois[active].length) { rois.splice(active, 1); active = rois.length - 1; } }
    refresh();
  });
  $("#roi-del").addEventListener("click", () => {
    if (active >= 0) { rois.splice(active, 1); active = rois.length - 1; }
    refresh();
  });
  $("#roi-reset").addEventListener("click", () => { rois = []; active = -1; refresh(); });
  $("#roi-cam").addEventListener("change", () => {
    const cam = camerasCache.find(c => c.id === $("#roi-cam").value);
    rois = cam && cam.rois ? cam.rois.map(r => r.map(p => [...p])) : [];
    active = rois.length ? 0 : -1;
    img = null; frameSize = cam && cam.frame_size ? cam.frame_size : null;
    if (frameSize) canvas.height = Math.round(canvas.width * frameSize[1] / frameSize[0]);
    refresh();
  });
  $("#roi-save").addEventListener("click", async () => {
    try {
      await api(`/cameras/${$("#roi-cam").value}/roi`, { method: "PUT", body: { rois, frame_size: frameSize } });
      await loadCameras();
      toast(`${rois.length} ROI${rois.length === 1 ? "" : "s"} saved`);
    } catch (e) { /* toast shown */ }
  });

  tabs.roi.onShow = async () => {
    const cur = $("#roi-cam").value;
    const cams = await loadCameras();
    $("#roi-cam").replaceChildren(...cams.map(c => h("option", { value: c.id }, c.name)));
    if (cams.some(c => c.id === cur)) $("#roi-cam").value = cur;
    $("#roi-cam").dispatchEvent(new Event("change"));
  };
  refresh();
})();
