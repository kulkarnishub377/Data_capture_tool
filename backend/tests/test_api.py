from datetime import datetime

import numpy as np
import pytest

import app as appmod
import settings
import storage


class FakeWorker:
    instances = []

    def __init__(self, camera, cfg, device, classes=None, model_name="", **kw):
        self.camera, self.device, self.classes, self.model_name = camera, device, classes, model_name
        self.running = False
        FakeWorker.instances.append(self)

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def status(self):
        return {"state": "connected" if self.running else "stopped", "saved": getattr(self, "saved", 0), "skipped": 0,
                "message": "", "device": self.device}

    def latest_jpeg(self):
        return b"\xff\xd8jpeg"


@pytest.fixture
def env(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "a.pt").write_bytes(b"x")
    (tmp_path / "models" / "b.pt").write_bytes(b"x")
    cfg = tmp_path / "config.ini"
    settings.save(cfg, {"MODELS_DIR": "models", "OUTPUT_DIR": "captures", "MODEL": "a.pt"})
    FakeWorker.instances.clear()
    frame = np.full((480, 640, 3), 100, dtype="uint8")
    app = appmod.create_app(cfg, tmp_path / "data", worker_factory=FakeWorker,
                            names_loader=lambda p: {0: "car", 1: "bus", 2: "truck"},
                            device_detector=lambda: [], grabber=lambda url: frame)
    return app.test_client(), tmp_path


def _cam(c, name="Cam-01"):
    return c.post("/api/cameras", json={"name": name, "url": "rtsp://x"}).get_json()


def test_camera_crud(env):
    c, _ = env
    cam = _cam(c)
    assert c.get("/api/cameras").get_json()[0]["id"] == cam["id"]
    assert c.put(f"/api/cameras/{cam['id']}", json={"name": "Renamed"}).status_code == 200
    assert c.get("/api/cameras").get_json()[0]["name"] == "Renamed"
    assert c.delete(f"/api/cameras/{cam['id']}").status_code == 200
    assert c.get("/api/cameras").get_json() == []


def test_add_camera_requires_name_and_url(env):
    c, _ = env
    assert c.post("/api/cameras", json={"name": "", "url": ""}).status_code == 400


def test_roi_needs_three_points(env):
    c, _ = env
    cam = _cam(c)
    bad = c.put(f"/api/cameras/{cam['id']}/roi", json={"points": [[0, 0], [1, 1]], "frame_size": [640, 480]})
    ok = c.put(f"/api/cameras/{cam['id']}/roi", json={"points": [[0, 0], [9, 0], [9, 9]], "frame_size": [640, 480]})
    assert bad.status_code == 400 and ok.status_code == 200


def test_snapshot_returns_jpeg_and_size_header(env):
    c, _ = env
    cam = _cam(c)
    r = c.get(f"/api/cameras/{cam['id']}/snapshot")
    assert r.status_code == 200 and r.mimetype == "image/jpeg"
    assert r.headers["X-Frame-Size"] == "640x480"


def test_models_lists_pt_files(env):
    c, _ = env
    assert c.get("/api/models").get_json() == ["a.pt", "b.pt"]


def test_model_classes_and_defaults(env):
    c, _ = env
    r = c.get("/api/model/classes").get_json()
    assert r["names"] == ["car", "bus", "truck"] and r["selected"] == ["car", "bus", "truck"]


def test_settings_validation_and_persistence(env):
    c, _ = env
    assert c.put("/api/settings", json={"SAVE_MODE": "nope"}).status_code == 400
    assert c.put("/api/settings", json={"MODEL": "missing.pt"}).status_code == 400
    assert c.put("/api/settings", json={"SAVE_METADATA": True, "SAVE_MODE": "both"}).status_code == 200
    s = c.get("/api/settings").get_json()
    assert s["SAVE_METADATA"] is True and s["SAVE_MODE"] == "both"


def test_devices_endpoint(env):
    c, _ = env
    r = c.get("/api/devices").get_json()
    assert r["pool"] == ["cpu"] and r["gpus"] == []


def test_start_requires_roi_and_model(env):
    c, tmp = env
    cam = _cam(c)
    r = c.post(f"/api/capture/{cam['id']}/start")
    assert r.status_code == 400 and "ROI" in r.get_json()["error"]
    c.put(f"/api/cameras/{cam['id']}/roi", json={"points": [[0, 0], [9, 0], [9, 9]], "frame_size": [640, 480]})
    (tmp / "models" / "a.pt").unlink()
    r = c.post(f"/api/capture/{cam['id']}/start")
    assert r.status_code == 400 and "model" in r.get_json()["error"].lower()


def test_start_stop_status(env):
    c, _ = env
    cam = _cam(c)
    c.put(f"/api/cameras/{cam['id']}/roi", json={"points": [[0, 0], [9, 0], [9, 9]], "frame_size": [640, 480]})
    assert c.post(f"/api/capture/{cam['id']}/start").status_code == 200
    assert c.get("/api/capture/status").get_json()[cam["id"]]["state"] == "connected"
    assert FakeWorker.instances[0].classes == ["car", "bus", "truck"]
    c.post(f"/api/capture/{cam['id']}/stop")
    assert c.get("/api/capture/status").get_json()[cam["id"]]["state"] == "stopped"


def test_unknown_selected_class_is_ignored(env):
    c, _ = env
    c.put("/api/settings", json={"CLASSES": ["car", "spaceship"]})
    cam = _cam(c)
    c.put(f"/api/cameras/{cam['id']}/roi", json={"points": [[0, 0], [9, 0], [9, 9]], "frame_size": [640, 480]})
    c.post(f"/api/capture/{cam['id']}/start")
    assert FakeWorker.instances[0].classes == ["car"]


def _make_capture(tmp, meta=None):
    f = np.full((100, 100, 3), 90, dtype="uint8")
    return storage.save_capture(tmp / "captures", "Cam-01", "car", datetime(2026, 11, 2, 9, 0, 0, 5000),
                                f, (10, 10, 50, 50), "frame", 90, 0.1, 0.4, meta)[0]


def test_gallery_list_serve_delete(env):
    c, tmp = env
    p = _make_capture(tmp, meta={"a": 1})
    rel = p.relative_to(tmp / "captures").as_posix()
    assert c.get("/api/captures").get_json()["total"] == 1
    assert c.get(f"/api/captures/{rel}").status_code == 200
    assert c.delete(f"/api/captures/{rel}").status_code == 200
    assert not p.exists() and not p.with_suffix(".json").exists()


@pytest.mark.parametrize("bad", ["..%2Fconfig.ini", "%2e%2e/config.ini", "Cam-01%5C..%5Cx.jpg"])
def test_capture_path_traversal_rejected(env, bad):
    c, _ = env
    assert c.get(f"/api/captures/{bad}").status_code in (400, 404)
    assert c.delete(f"/api/captures/{bad}").status_code in (400, 404)


def test_stream_404_when_not_running(env):
    c, _ = env
    assert c.get("/api/stream/nope").status_code == 404


def test_api_never_returns_camera_password(env):
    c, _ = env
    r = c.post("/api/cameras", json={"name": "S", "url": "rtsp://admin:Secret@123@1.2.3.4:554/s"}).get_json()
    assert "Secret" not in r["url"] and "****" in r["url"]
    assert "Secret" not in c.get("/api/cameras").get_data(as_text=True)


def _ready_cam(c):
    cam = _cam(c)
    c.put(f"/api/cameras/{cam['id']}/roi", json={"points": [[0, 0], [9, 0], [9, 9]], "frame_size": [640, 480]})
    return cam


def test_disabling_running_camera_stops_it(env):
    c, _ = env
    cam = _ready_cam(c)
    c.post(f"/api/capture/{cam['id']}/start")
    c.put(f"/api/cameras/{cam['id']}", json={"enabled": False})
    assert FakeWorker.instances[0].running is False


def test_start_replaces_worker_in_error_state(env):
    c, _ = env
    cam = _ready_cam(c)
    c.post(f"/api/capture/{cam['id']}/start")
    FakeWorker.instances[0].status = lambda: {"state": "error", "saved": 0, "skipped": 0,
                                              "message": "x", "device": "cpu"}
    r = c.post(f"/api/capture/{cam['id']}/start")
    assert r.status_code == 200 and len(FakeWorker.instances) == 2


def test_unknown_class_names_produce_visible_warning(env):
    c, _ = env
    c.put("/api/settings", json={"CLASSES": ["car", "spaceship"]})
    cam = _ready_cam(c)
    r = c.post(f"/api/capture/{cam['id']}/start").get_json()
    assert any("spaceship" in w for w in r["warnings"])
    assert any("spaceship" in w for w in c.get("/api/model/classes").get_json()["warnings"])


def test_snapshot_failure_gives_clear_message(tmp_path):
    (tmp_path / "models").mkdir()
    cfg = tmp_path / "config.ini"
    settings.save(cfg, {"MODELS_DIR": "models", "OUTPUT_DIR": "captures"})
    c = appmod.create_app(cfg, tmp_path / "data", worker_factory=FakeWorker, grabber=lambda u: None).test_client()
    cam = c.post("/api/cameras", json={"name": "A", "url": "rtsp://x"}).get_json()
    r = c.get(f"/api/cameras/{cam['id']}/snapshot")
    assert r.status_code == 502 and "respond" in r.get_json()["error"]
    t = c.post(f"/api/cameras/{cam['id']}/test").get_json()
    assert t["ok"] is False and "respond" in t["error"]


def test_snapshot_surfaces_unreachable_camera_reason(tmp_path):
    (tmp_path / "models").mkdir()
    cfg = tmp_path / "config.ini"
    settings.save(cfg, {"MODELS_DIR": "models", "OUTPUT_DIR": "captures"})

    def grab(url):
        raise appmod.CameraError("Cannot reach 1.2.3.4:5549 (timed out) - check host and port")

    c = appmod.create_app(cfg, tmp_path / "data", worker_factory=FakeWorker, grabber=grab).test_client()
    cam = c.post("/api/cameras", json={"name": "A", "url": "rtsp://1.2.3.4:5549/x"}).get_json()
    r = c.get(f"/api/cameras/{cam['id']}/snapshot")
    assert r.status_code == 502 and "1.2.3.4:5549" in r.get_json()["error"]


def test_multiple_rois_saved_and_returned(env):
    c, _ = env
    cam = _cam(c)
    tri = [[0, 0], [9, 0], [9, 9]]
    r = c.put(f"/api/cameras/{cam['id']}/roi", json={"rois": [tri, tri], "frame_size": [640, 480]})
    assert r.status_code == 200
    assert len(c.get("/api/cameras").get_json()[0]["rois"]) == 2


def test_rois_with_an_invalid_polygon_rejected(env):
    c, _ = env
    cam = _cam(c)
    tri = [[0, 0], [9, 0], [9, 9]]
    r = c.put(f"/api/cameras/{cam['id']}/roi", json={"rois": [tri, [[0, 0], [1, 1]]], "frame_size": [640, 480]})
    assert r.status_code == 400


import time as _time


def _wait(cond, timeout=3.0):
    t = _time.time()
    while _time.time() - t < timeout:
        if cond():
            return True
        _time.sleep(0.02)
    return False


def _running_cam(c):
    cam = _ready_cam(c)
    assert c.post(f"/api/capture/{cam['id']}/start").status_code == 200
    return cam


def test_changing_model_restarts_running_cameras_and_keeps_counts(env):
    c, _ = env
    cam = _running_cam(c)
    FakeWorker.instances[0].saved = 5
    assert c.put("/api/settings", json={"MODEL": "b.pt", "CLASSES": []}).status_code == 200
    assert _wait(lambda: len(FakeWorker.instances) == 2)
    old, new = FakeWorker.instances
    assert old.running is False and new.running is True and new.model_name == "b.pt"
    assert c.get("/api/capture/status").get_json()[cam["id"]]["saved"] == 5  # carried over


def test_threshold_change_restarts_but_identical_value_does_not(env):
    c, _ = env
    _running_cam(c)
    c.put("/api/settings", json={"CONF_THRESH": 0.25})          # same as current
    _time.sleep(0.3)
    assert len(FakeWorker.instances) == 1
    c.put("/api/settings", json={"CONF_THRESH": 0.6})
    assert _wait(lambda: len(FakeWorker.instances) == 2)
    assert FakeWorker.instances[1].cfg["CONF_THRESH"] == 0.6


def test_settings_change_while_stopped_starts_nothing(env):
    c, _ = env
    _ready_cam(c)
    c.put("/api/settings", json={"CONF_THRESH": 0.6})
    _time.sleep(0.3)
    assert FakeWorker.instances == []


def test_out_of_range_threshold_rejected(env):
    c, _ = env
    assert c.put("/api/settings", json={"CONF_THRESH": 5}).status_code == 400
    assert c.put("/api/settings", json={"CONF_THRESH": "abc"}).status_code == 400


def test_recent_and_thumb_endpoints(env):
    c, tmp = env
    p = _make_capture(tmp)
    rel = p.relative_to(tmp / "captures").as_posix()
    r = c.get("/api/recent?limit=5").get_json()
    assert len(r["items"]) == 1 and r["items"][0]["class"] == "car"
    t = c.get(f"/api/thumb/{rel}?w=64")
    assert t.status_code == 200 and t.mimetype == "image/jpeg"
    assert c.get("/api/thumb/..%2Fconfig.ini").status_code in (400, 404)


def test_status_reports_restarting_state_during_restart(env):
    c, _ = env
    cam = _running_cam(c)
    FakeWorker.instances[0].stop = lambda: _time.sleep(0.4)   # slow shutdown keeps the restart in flight
    c.put("/api/settings", json={"CONF_THRESH": 0.7})
    assert _wait(lambda: c.get("/api/capture/status").get_json()[cam["id"]]["state"] == "restarting")
