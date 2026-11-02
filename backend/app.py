"""Data Capture Tool - Flask backend (REST API + MJPEG preview).

Run:  python backend/app.py        (serves the frontend/ folder; host and port come from config.ini [App])
"""
import logging
import os
import sys
import socket
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent))

import camera_io  # noqa: E402
import capture_engine  # noqa: E402
import devices  # noqa: E402
import settings  # noqa: E402
import storage  # noqa: E402

from camera_io import CameraError, check_port, grab_frame, run_with_timeout  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("app")


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
    carry = {}         # cid -> counters of workers replaced by a restart (so the UI counts keep growing)
    restarting = set()  # cids whose worker is being replaced right now
    failed = {}        # cid -> why an automatic restart could not bring the camera back
    WORKER_KEYS = {"MODEL", "DEVICE", "HALF", "CONF_THRESH", "IMG_SIZE", "CLASSES", "SAVE_MODE",
                   "SAVE_METADATA", "WHEN", "MIN_BOX", "MIN_ROI_OVERLAP", "JPEG_QUALITY",
                   "PAD_NORMAL", "PAD_BIKE", "DEDUP_SECONDS", "MAX_HOLD_SECONDS"}

    def err(msg, code=400):
        return jsonify({"error": msg}), code

    def public(cam):
        """Camera as sent to the browser: the password never leaves the server."""
        out = dict(cam)
        out["url"] = storage.mask_url(out.get("url", ""))
        return out

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

    def class_selection(c, names):
        """(classes to capture, warnings). Unknown configured names are reported, never silent."""
        known = set(names.values())
        chosen = [n for n in c["CLASSES"] if n in known]
        warns = [f"Class '{n}' is not in model {c['MODEL']} - ignored" for n in c["CLASSES"] if n not in known]
        if not chosen:
            if c["CLASSES"]:
                warns.append("None of the selected classes exist in this model - using its default classes")
            chosen = capture_engine.default_classes(names)
        return chosen, warns

    # ---- frontend ---------------------------------------------------------
    @app.get("/")
    def index():
        return send_from_directory(frontend_dir, "index.html")

    # ---- cameras ----------------------------------------------------------
    @app.get("/api/cameras")
    def cameras_list():
        return jsonify([public(c) for c in store.list()])

    @app.post("/api/cameras")
    def cameras_add():
        d = request.get_json(silent=True) or {}
        name, url = str(d.get("name", "")).strip(), str(d.get("url", "")).strip()
        if not name or not url:
            return err("name and url are required")
        return jsonify(public(store.add(name, url, device=d.get("device") or None)))

    @app.put("/api/cameras/<cid>")
    def cameras_update(cid):
        d = request.get_json(silent=True) or {}
        fields = {k: d[k] for k in ("name", "url", "enabled", "device") if k in d}
        if "****" in str(fields.get("url", "")):
            fields.pop("url")  # a masked URL echoed back by a client must not overwrite the real one
        try:
            cam = store.update(cid, **fields)
        except KeyError:
            return err("camera not found", 404)
        if fields.get("enabled") is False:
            with wlock:
                w = workers.get(cid)
            if w:
                w.stop()
        return jsonify(public(cam))

    @app.delete("/api/cameras/<cid>")
    def cameras_delete(cid):
        with wlock:
            w = workers.pop(cid, None)
        if w:
            w.stop()
        carry.pop(cid, None)
        failed.pop(cid, None)
        store.delete(cid)
        return jsonify({"ok": True})

    @app.post("/api/cameras/<cid>/test")
    def cameras_test(cid):
        cam = store.get(cid)
        if not cam:
            return err("camera not found", 404)
        try:
            frame = grabber(cam["url"])
        except CameraError as e:
            return jsonify({"ok": False, "error": str(e)})
        if frame is None:
            return jsonify({"ok": False, "error": "The camera did not respond with a picture"})
        return jsonify({"ok": True, "size": [frame.shape[1], frame.shape[0]]})

    @app.get("/api/cameras/<cid>/snapshot")
    def cameras_snapshot(cid):
        cam = store.get(cid)
        if not cam:
            return err("camera not found", 404)
        try:
            frame = grabber(cam["url"])
        except CameraError as e:
            return err(str(e), 502)
        if frame is None:
            return err("The camera did not respond with a picture", 502)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        r = Response(buf.tobytes(), mimetype="image/jpeg")
        r.headers["X-Frame-Size"] = f"{frame.shape[1]}x{frame.shape[0]}"
        r.headers["Access-Control-Expose-Headers"] = "X-Frame-Size"
        return r

    @app.put("/api/cameras/<cid>/roi")
    def cameras_roi(cid):
        d = request.get_json(silent=True) or {}
        try:
            rois = d["rois"] if "rois" in d else [d.get("points", [])]
            return jsonify(public(store.set_rois(cid, rois, d.get("frame_size") or [0, 0])))
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
        before = settings.load(config_path)
        try:
            settings.save(config_path, d)
        except (ValueError, TypeError) as e:
            return err(str(e))
        after = settings.load(config_path)
        if any(k in WORKER_KEYS and before.get(k) != after.get(k) for k in d):
            restart_running()
        return jsonify(after)

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
        selected, warns = class_selection(dict(c, MODEL=model), names)
        return jsonify({"names": list(names.values()), "selected": selected, "warnings": warns,
                        "default": capture_engine.default_classes(names)})

    # ---- capture control --------------------------------------------------
    def start_worker(cid):
        """Build + start the worker for a camera. Returns (payload dict, http status)."""
        cam = store.get(cid)
        if not cam:
            return {"error": "camera not found"}, 404
        if not cam.get("rois"):
            return {"error": "Draw an ROI (at least 3 points) before starting capture"}, 400
        c = cfg()
        if not (Path(c["MODELS_DIR"]) / c["MODEL"]).is_file():
            return {"error": f"Model file not found: {c['MODEL']}"}, 400
        try:
            names = model_names(c, c["MODEL"])
        except Exception as e:  # noqa: BLE001
            return {"error": f"Could not load model: {e}"}, 400
        with wlock:
            old = workers.get(cid)
            if old and old.status()["state"] not in ("stopped", "error"):
                return {"ok": True, "already_running": True}, 200
            if old:
                old.stop()  # release the old worker's threads, keep its counts
                s = old.status()
                b = carry.setdefault(cid, {"saved": 0, "skipped": 0})
                b["saved"] += s["saved"]
                b["skipped"] += s["skipped"]
            pool, warns = devices.resolve(c["DEVICE"], [g["id"] for g in device_detector()])
            cams = store.list()
            assign = devices.assign([x["id"] for x in cams], pool,
                                    {x["id"]: x.get("device") for x in cams if x.get("device")})
            classes, class_warns = class_selection(c, names)
            w = worker_factory(cam, c, assign[cid], classes=classes, model_name=c["MODEL"])
            workers[cid] = w
            failed.pop(cid, None)
            w.start()
        return {"ok": True, "device": assign[cid], "warnings": warns + class_warns}, 200

    def restart_one(cid):
        """Apply new settings to a running camera. Saved images and counters are kept."""
        try:
            with wlock:
                old = workers.get(cid)
            if old:
                old.stop()
            payload, code = start_worker(cid)
            if code != 200:
                failed[cid] = payload.get("error", "restart failed")
            elif payload.get("already_running"):
                failed[cid] = "The previous capture is still shutting down - press Start again"
        except Exception as e:  # noqa: BLE001
            log.exception("restart failed")
            failed[cid] = str(e)
        finally:
            restarting.discard(cid)

    def restart_running():
        with wlock:
            ids = [cid for cid, w in workers.items() if w.status()["state"] not in ("stopped", "error")]
            restarting.update(ids)
        for cid in ids:
            threading.Thread(target=restart_one, args=(cid,), daemon=True).start()

    @app.post("/api/capture/<cid>/start")
    def capture_start(cid):
        payload, code = start_worker(cid)
        return jsonify(payload), code

    @app.post("/api/capture/<cid>/stop")
    def capture_stop(cid):
        with wlock:
            w = workers.get(cid)
        if w:
            w.stop()
        return jsonify({"ok": True})

    @app.post("/api/capture/start-all")
    def capture_start_all():
        """Start every enabled camera. Cameras that cannot start (no ROI, bad model...) are reported, not fatal."""
        results = {}
        counts = {"started": 0, "failed": 0, "skipped": 0, "already_running": 0}
        for cam in store.list():
            cid = cam["id"]
            if cam.get("enabled") is False:
                results[cid] = {"skipped": "disabled", "name": cam["name"]}
                counts["skipped"] += 1
                continue
            payload, code = start_worker(cid)
            results[cid] = dict(payload, name=cam["name"])
            if code != 200:
                counts["failed"] += 1
            elif payload.get("already_running"):
                counts["already_running"] += 1
            else:
                counts["started"] += 1
        return jsonify(dict(counts, results=results))

    @app.post("/api/capture/stop-all")
    def capture_stop_all():
        with wlock:
            running = [w for w in workers.values() if w.status()["state"] not in ("stopped", "error")]
        threads = [threading.Thread(target=w.stop, daemon=True) for w in running]  # stop in parallel
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        return jsonify({"stopped": len(running)})

    @app.get("/api/capture/status")
    def capture_status():
        with wlock:
            snapshot = dict(workers)
        out = {}
        for cid, w in snapshot.items():
            st = w.status()
            b = carry.get(cid)
            if b:
                st["saved"] += b["saved"]
                st["skipped"] += b["skipped"]
            if cid in restarting:
                st["state"], st["message"] = "restarting", "applying new settings"
            elif cid in failed and st["state"] in ("stopped", "error"):
                st["state"], st["message"] = "error", failed[cid]
            out[cid] = st
        return jsonify(out)

    @app.get("/api/stream/<cid>")
    def stream(cid):
        with wlock:
            if workers.get(cid) is None:
                return err("not running", 404)

        def gen():
            last = None
            while True:
                with wlock:
                    w = workers.get(cid)   # re-read each loop: the stream survives a restart
                if w is None:
                    return
                jpg = w.latest_jpeg()
                if jpg is not None and jpg is not last:
                    last = jpg
                    yield b"--frame%sContent-Type: image/jpeg%s%s" % (b"\r\n", b"\r\n\r\n", jpg) + b"\r\n"
                elif w.status()["state"] == "stopped" and cid not in restarting:
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

    @app.get("/api/recent")
    def recent():
        try:
            limit = min(100, max(1, int(request.args.get("limit", 10))))
        except ValueError:
            return err("limit must be a number")
        return jsonify({"items": storage.recent_captures(cfg()["OUTPUT_DIR"], limit)})

    @app.get("/api/thumb/<path:rel>")
    def thumb(rel):
        try:
            p = storage.make_thumbnail(cfg()["OUTPUT_DIR"], rel, data_dir / "thumbs",
                                       int(request.args.get("w", 320)))
        except ValueError as e:
            return err(str(e))
        except FileNotFoundError:
            abort(404)
        r = send_file(p, mimetype="image/jpeg")
        r.headers["Cache-Control"] = "public, max-age=3600"
        return r

    app.workers = workers
    return app


LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def server_options(cfg, env) -> dict:
    """Where and how to serve. config.ini [App] HOST/PORT, optionally overridden by the HOST / PORT env vars.

    The interactive debugger is a remote-code-execution console, so it is only ever enabled when the server is
    reachable from this machine alone (loopback). DEBUG logging stays on either way."""
    host = env.get("HOST") or cfg["HOST"]
    try:
        port = int(env.get("PORT") or cfg["PORT"])
    except ValueError:
        raise ValueError(f"PORT must be a number, got {env.get('PORT')!r}") from None
    if not 1 <= port <= 65535:
        raise ValueError(f"PORT must be between 1 and 65535, got {port}")
    return {"host": host, "port": port, "debug": bool(cfg["DEBUG"]),
            "use_debugger": bool(cfg["DEBUG"]) and host in LOOPBACK_HOSTS, "use_reloader": False}


def lan_addresses() -> list:
    """This machine's non-loopback IPv4 addresses (what other machines would type to reach it)."""
    found = set()
    try:
        found.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    try:  # the address of the interface that routes outwards (no packet is sent)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            found.add(s.getsockname()[0])
    except OSError:
        pass
    return sorted(ip for ip in found if ip.count(".") == 3 and not ip.startswith("127."))


def main():
    config_path = os.environ.get("DCT_CONFIG") or ROOT / "config.ini"
    cfg = settings.load(config_path)
    opts = server_options(cfg, os.environ)
    logging.basicConfig(level=logging.DEBUG if opts["debug"] else logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    app = create_app(config_path, os.environ.get("DCT_DATA"))
    # importing torch takes seconds: do it now so the first page load is quick
    threading.Thread(target=devices.detect, daemon=True).start()

    host, port = opts["host"], opts["port"]
    print(f"Data Capture Tool listening on {host}:{port}  (debug={'on' if opts['debug'] else 'off'})")
    print(f"  this machine : http://127.0.0.1:{port}")
    if host in ("0.0.0.0", "::"):
        for ip in lan_addresses():
            print(f"  other machines: http://{ip}:{port}")
        print("  (no login: expose it on a trusted network only; allow the port in the firewall)")
    if opts["debug"] and not opts["use_debugger"]:
        print("  interactive debugger is OFF because the server is reachable from the network (DEBUG logs stay on)")
    # use_reloader=False: the reloader would restart the process and kill running captures
    app.run(host=host, port=port, threaded=True, debug=opts["debug"],
            use_debugger=opts["use_debugger"], use_reloader=opts["use_reloader"])


if __name__ == "__main__":
    main()
