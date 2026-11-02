"""Data Capture Tool - Flask backend (REST API + MJPEG preview).

Run:  python backend/app.py        (serves the frontend/ folder at http://127.0.0.1:5100)
"""
import logging
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

import cv2
from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent))

import capture_engine  # noqa: E402
import devices  # noqa: E402
import settings  # noqa: E402
import storage  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("app")


def grab_frame(url):
    """Grab one frame from a stream URL (or file); None on failure."""
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|analyzeduration;3000000|probesize;3000000|stimeout;5000000")
    cap = cv2.VideoCapture(str(url), cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def create_app(config_path=None, data_dir=None, frontend_dir=None,
               worker_factory=capture_engine.CameraWorker,
               names_loader=capture_engine.load_model_names,
               device_detector=devices.detect, grabber=grab_frame):
    config_path = Path(config_path or ROOT / "config.ini")
    data_dir = Path(data_dir or ROOT / "data")
    frontend_dir = Path(frontend_dir or ROOT / "frontend")
    app = Flask(__name__, static_folder=str(frontend_dir / "static"), static_url_path="/static")
    store = storage.CameraStore(data_dir / "cameras.json")
    workers, wlock = {}, threading.Lock()
    names_cache = {}

    def err(msg, code=400):
        return jsonify({"error": msg}), code

    def cfg():
        c = settings.load(config_path)
        base = config_path.parent
        for k in ("MODELS_DIR", "OUTPUT_DIR"):
            c[k] = str((base / c[k]).resolve())
        return c

    def model_names(c, model):
        p = Path(c["MODELS_DIR"]) / model
        key = (str(p), p.stat().st_mtime)
        if key not in names_cache:
            names_cache[key] = names_loader(p)
        return names_cache[key]

    def selected_classes(c, names):
        chosen = [n for n in c["CLASSES"] if n in names.values()]
        return chosen or capture_engine.default_classes(names)

    # ---- frontend ---------------------------------------------------------
    @app.get("/")
    def index():
        return send_from_directory(frontend_dir, "index.html")

    # ---- cameras ----------------------------------------------------------
    @app.get("/api/cameras")
    def cameras_list():
        return jsonify(store.list())

    @app.post("/api/cameras")
    def cameras_add():
        d = request.get_json(silent=True) or {}
        name, url = str(d.get("name", "")).strip(), str(d.get("url", "")).strip()
        if not name or not url:
            return err("name and url are required")
        return jsonify(store.add(name, url, device=d.get("device") or None))

    @app.put("/api/cameras/<cid>")
    def cameras_update(cid):
        d = request.get_json(silent=True) or {}
        fields = {k: d[k] for k in ("name", "url", "enabled", "device") if k in d}
        try:
            return jsonify(store.update(cid, **fields))
        except KeyError:
            return err("camera not found", 404)

    @app.delete("/api/cameras/<cid>")
    def cameras_delete(cid):
        with wlock:
            w = workers.pop(cid, None)
        if w:
            w.stop()
        store.delete(cid)
        return jsonify({"ok": True})

    @app.post("/api/cameras/<cid>/test")
    def cameras_test(cid):
        cam = store.get(cid)
        if not cam:
            return err("camera not found", 404)
        frame = grabber(cam["url"])
        if frame is None:
            return jsonify({"ok": False, "error": "could not read a frame"})
        return jsonify({"ok": True, "size": [frame.shape[1], frame.shape[0]]})

    @app.get("/api/cameras/<cid>/snapshot")
    def cameras_snapshot(cid):
        cam = store.get(cid)
        if not cam:
            return err("camera not found", 404)
        frame = grabber(cam["url"])
        if frame is None:
            return err("could not read a frame from the camera", 502)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        r = Response(buf.tobytes(), mimetype="image/jpeg")
        r.headers["X-Frame-Size"] = f"{frame.shape[1]}x{frame.shape[0]}"
        r.headers["Access-Control-Expose-Headers"] = "X-Frame-Size"
        return r

    @app.put("/api/cameras/<cid>/roi")
    def cameras_roi(cid):
        d = request.get_json(silent=True) or {}
        try:
            return jsonify(store.set_roi(cid, d.get("points", []), d.get("frame_size") or [0, 0]))
        except ValueError as e:
            return err(str(e))
        except KeyError:
            return err("camera not found", 404)

    # ---- settings / models / devices -------------------------------------
    @app.get("/api/models")
    def models_list():
        d = Path(cfg()["MODELS_DIR"])
        return jsonify(sorted(p.name for p in d.glob("*.pt")) if d.is_dir() else [])

    @app.get("/api/devices")
    def devices_list():
        gpus = device_detector()
        pool, warns = devices.resolve(cfg()["DEVICE"], [g["id"] for g in gpus])
        return jsonify({"gpus": gpus, "pool": pool, "warnings": warns})

    @app.get("/api/settings")
    def settings_get():
        c = settings.load(config_path)
        return jsonify(c)

    @app.put("/api/settings")
    def settings_put():
        d = request.get_json(silent=True) or {}
        if "MODEL" in d:
            m = str(d["MODEL"])
            if Path(m).name != m or not (Path(cfg()["MODELS_DIR"]) / m).is_file():
                return err(f"Model file not found: {m}")
        try:
            settings.save(config_path, d)
        except (ValueError, TypeError) as e:
            return err(str(e))
        return jsonify(settings.load(config_path))

    @app.get("/api/model/classes")
    def model_classes():
        c = cfg()
        model = request.args.get("model") or c["MODEL"]
        if Path(model).name != model or not (Path(c["MODELS_DIR"]) / model).is_file():
            return err(f"Model file not found: {model}")
        try:
            names = model_names(c, model)
        except Exception as e:  # noqa: BLE001
            return err(f"Could not load model: {e}")
        return jsonify({"names": list(names.values()), "selected": selected_classes(c, names)})

    # ---- capture control --------------------------------------------------
    @app.post("/api/capture/<cid>/start")
    def capture_start(cid):
        cam = store.get(cid)
        if not cam:
            return err("camera not found", 404)
        if len(cam.get("roi") or []) < 3:
            return err("Draw an ROI (at least 3 points) before starting capture")
        c = cfg()
        mp = Path(c["MODELS_DIR"]) / c["MODEL"]
        if not mp.is_file():
            return err(f"Model file not found: {c['MODEL']}")
        try:
            names = model_names(c, c["MODEL"])
        except Exception as e:  # noqa: BLE001
            return err(f"Could not load model: {e}")
        with wlock:
            old = workers.get(cid)
            if old and old.status()["state"] != "stopped":
                return jsonify({"ok": True, "already_running": True})
            pool, warns = devices.resolve(c["DEVICE"], [g["id"] for g in device_detector()])
            cams = store.list()
            assign = devices.assign([x["id"] for x in cams], pool,
                                    {x["id"]: x.get("device") for x in cams if x.get("device")})
            w = worker_factory(cam, c, assign[cid], classes=selected_classes(c, names),
                               model_name=c["MODEL"])
            workers[cid] = w
            w.start()
        return jsonify({"ok": True, "device": assign[cid], "warnings": warns})

    @app.post("/api/capture/<cid>/stop")
    def capture_stop(cid):
        with wlock:
            w = workers.get(cid)
        if w:
            w.stop()
        return jsonify({"ok": True})

    @app.get("/api/capture/status")
    def capture_status():
        with wlock:
            return jsonify({k: w.status() for k, w in workers.items()})

    @app.get("/api/stream/<cid>")
    def stream(cid):
        with wlock:
            w = workers.get(cid)
        if w is None:
            return err("not running", 404)

        def gen():
            last = None
            while True:
                jpg = w.latest_jpeg()
                if jpg is not None and jpg is not last:
                    last = jpg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                elif w.status()["state"] == "stopped":
                    return
                time.sleep(0.08)

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    # ---- gallery ----------------------------------------------------------
    @app.get("/api/captures")
    def captures_list():
        a = request.args
        return jsonify(storage.list_captures(
            cfg()["OUTPUT_DIR"], camera=a.get("camera") or None, date=a.get("date") or None,
            cls=a.get("class") or None, page=max(1, int(a.get("page", 1) or 1)),
            per_page=min(200, int(a.get("per_page", 60) or 60))))

    @app.get("/api/captures/<path:rel>")
    def captures_get(rel):
        try:
            p = storage.safe_capture_path(cfg()["OUTPUT_DIR"], rel)
        except ValueError as e:
            return err(str(e))
        if p.suffix.lower() != ".jpg" or not p.is_file():
            abort(404)
        return send_file(p, mimetype="image/jpeg")

    @app.delete("/api/captures/<path:rel>")
    def captures_delete(rel):
        try:
            storage.delete_capture(cfg()["OUTPUT_DIR"], rel)
        except ValueError as e:
            return err(str(e))
        except FileNotFoundError:
            return err("not found", 404)
        return jsonify({"ok": True})

    app.workers = workers
    return app


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    port = int(os.environ.get("PORT", 5100))
    app = create_app()
    if os.environ.get("NO_BROWSER") != "1":
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    print(f"Data Capture Tool running at http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, threaded=True)


if __name__ == "__main__":
    main()
