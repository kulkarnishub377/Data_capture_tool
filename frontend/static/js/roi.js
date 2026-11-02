/* ROI tab: click to add points, drag to move, undo/reset/save. Points are in true frame pixels. */
(() => {
  const canvas = $("#roi-canvas");
  const ctx = canvas.getContext("2d");
  const R = 8;                // point hit radius (display px)
  let img = null;             // snapshot image
  let frameSize = null;       // [w, h] of the real camera frame
  let points = [];            // [[x, y]] in frame pixels
  let drag = -1;

  const scale = () => (frameSize ? canvas.width / frameSize[0] : 1);

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (img) ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    else {
      ctx.fillStyle = "#9aa3b2"; ctx.font = "16px system-ui";
      ctx.fillText("Choose a camera and click 'Load snapshot'", 20, 30);
    }
    const s = scale();
    if (points.length) {
      ctx.beginPath();
      points.forEach(([x, y], i) => (i ? ctx.lineTo(x * s, y * s) : ctx.moveTo(x * s, y * s)));
      if (points.length >= 3) { ctx.closePath(); ctx.fillStyle = "rgba(37,99,235,.25)"; ctx.fill(); }
      ctx.strokeStyle = "#facc15"; ctx.lineWidth = 2; ctx.stroke();
      points.forEach(([x, y], i) => {
        ctx.beginPath(); ctx.arc(x * s, y * s, R - 2, 0, 7);
        ctx.fillStyle = i === 0 ? "#22c55e" : "#facc15"; ctx.fill();
      });
    }
    $("#roi-save").disabled = !(points.length >= 3 && frameSize);
  }

  function pos(e) {
    const r = canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * (canvas.width / r.width), (e.clientY - r.top) * (canvas.height / r.height)];
  }
  const hit = ([cx, cy]) => {
    const s = scale();
    return points.findIndex(([x, y]) => Math.hypot(x * s - cx, y * s - cy) <= R + 4);
  };

  canvas.addEventListener("pointerdown", e => {
    if (!frameSize) { toast("Load a snapshot first", true); return; }
    const p = pos(e), i = hit(p);
    if (i >= 0) { drag = i; canvas.setPointerCapture(e.pointerId); }
    else { const s = scale(); points.push([Math.round(p[0] / s), Math.round(p[1] / s)]); draw(); }
  });
  canvas.addEventListener("pointermove", e => {
    if (drag < 0) return;
    const [cx, cy] = pos(e), s = scale();
    points[drag] = [Math.max(0, Math.min(frameSize[0], Math.round(cx / s))),
                    Math.max(0, Math.min(frameSize[1], Math.round(cy / s)))];
    draw();
  });
  canvas.addEventListener("pointerup", () => { drag = -1; });

  async function loadSnapshot() {
    const id = $("#roi-cam").value;
    if (!id) return toast("Add a camera first", true);
    toast("Grabbing a frame...");
    try {
      const res = await api(`/cameras/${id}/snapshot?t=${Date.now()}`, { raw: true });
      const m = /^(\d+)x(\d+)$/.exec(res.headers.get("X-Frame-Size") || "");
      const blob = await res.blob();
      const im = new Image();
      im.onload = () => {
        img = im;
        frameSize = m ? [+m[1], +m[2]] : [im.naturalWidth, im.naturalHeight];
        canvas.height = Math.round(canvas.width * frameSize[1] / frameSize[0]);
        draw(); toast("Snapshot loaded");
      };
      im.src = URL.createObjectURL(blob);
    } catch (e) { /* toast shown */ }
  }

  $("#roi-load").addEventListener("click", loadSnapshot);
  $("#roi-undo").addEventListener("click", () => { points.pop(); draw(); });
  $("#roi-reset").addEventListener("click", () => { points = []; draw(); });
  $("#roi-cam").addEventListener("change", () => {
    const cam = camerasCache.find(c => c.id === $("#roi-cam").value);
    points = cam && cam.roi ? cam.roi.map(p => [...p]) : [];
    img = null; frameSize = cam && cam.frame_size ? cam.frame_size : null;
    if (frameSize) canvas.height = Math.round(canvas.width * frameSize[1] / frameSize[0]);
    draw();
  });
  $("#roi-save").addEventListener("click", async () => {
    try {
      await api(`/cameras/${$("#roi-cam").value}/roi`, { method: "PUT", body: { points, frame_size: frameSize } });
      await loadCameras();
      toast("ROI saved");
    } catch (e) { /* toast shown */ }
  });

  tabs.roi.onShow = async () => {
    const cur = $("#roi-cam").value;
    const cams = await loadCameras();
    $("#roi-cam").replaceChildren(...cams.map(c => h("option", { value: c.id }, c.name)));
    if (cams.some(c => c.id === cur)) $("#roi-cam").value = cur;
    $("#roi-cam").dispatchEvent(new Event("change"));
  };
  draw();
})();
