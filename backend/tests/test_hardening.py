"""Robustness: input validation, thread safety of shared state, clean shutdown of streams and GPU access."""
import threading
import time

import numpy as np
import pytest

import app as appmod
import camera_io
import capture_engine as ce
import gpu_lock
import settings
from test_api import FakeWorker, _ready_cam, env  # noqa: F401  (fixture re-used)
from test_engine import CLASSES, _camera, _video, _wait_done, cfg, det


# ---------------------------------------------------------------- camera URL validation
@pytest.mark.parametrize("url", ["rtsp://admin:pw@10.0.0.5:554/stream1", "rtsps://cam.local/live",
                                 "rtsp://admin:Se#c/r@t@10.0.0.5:554/s",       # special chars in the password
                                 "rtsp://[fe80::1]:554/s",
                                 "http://10.0.0.9:8080/video", "rtmp://srv/app/key"])
def test_valid_camera_urls_are_accepted(url):
    assert camera_io.validate_camera_url(url) == url


def test_an_existing_video_file_is_accepted_as_a_source(tmp_path):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"x")
    assert camera_io.validate_camera_url(f"  {f}  ") == str(f)


@pytest.mark.parametrize("url", ["not a url", "rtsp://", "rtsp:///stream", "rtsp://host:99999/s",
                                 "ftp://host/file", "C:/no/such/video.mp4"])
def test_invalid_camera_urls_are_rejected_with_a_clear_message(url):
    with pytest.raises(ValueError, match="rtsp://"):
        camera_io.validate_camera_url(url)


def test_add_and_update_camera_reject_a_bad_url(env):  # noqa: F811
    c, _ = env
    r = c.post("/api/cameras", json={"name": "A", "url": "rtsp:/typo"})
    assert r.status_code == 400 and "rtsp://" in r.get_json()["error"]
    cam = c.post("/api/cameras", json={"name": "A", "url": "rtsp://10.0.0.5/s"}).get_json()
    assert c.put(f"/api/cameras/{cam['id']}", json={"url": "garbage"}).status_code == 400
    assert c.put(f"/api/cameras/{cam['id']}", json={"url": "rtsp://10.0.0.6/s"}).status_code == 200


# ---------------------------------------------------------------- bad query parameters -> 400, never 500
@pytest.mark.parametrize("path", ["/api/captures?page=abc", "/api/captures?per_page=x", "/api/thumb/a/b.jpg?w=wide"])
def test_non_numeric_query_parameters_give_400(env, path):  # noqa: F811
    c, _ = env
    r = c.get(path)
    assert r.status_code == 400 and "number" in r.get_json()["error"]


# ---------------------------------------------------------------- shared state in the app
def test_a_model_is_loaded_once_even_when_many_requests_ask_at_once(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "a.pt").write_bytes(b"x" * 20000)
    cfgp = tmp_path / "config.ini"
    settings.save(cfgp, {"MODELS_DIR": "models", "OUTPUT_DIR": "captures", "MODEL": "a.pt"})
    loads = []

    def slow_loader(p):
        loads.append(p)
        time.sleep(0.2)
        return {0: "car"}

    app = appmod.create_app(cfgp, tmp_path / "data", worker_factory=FakeWorker, names_loader=slow_loader,
                            device_detector=lambda: [], grabber=lambda url: None)
    codes = []
    threads = [threading.Thread(target=lambda: codes.append(app.test_client().get("/api/model/classes").status_code))
               for _ in range(5)]
    [t.start() for t in threads]
    [t.join(5) for t in threads]
    assert codes == [200] * 5 and len(loads) == 1


class ErrorWorker(FakeWorker):
    def status(self):
        return dict(super().status(), state="error", message="boom")


def test_preview_stream_ends_when_the_worker_has_failed(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "a.pt").write_bytes(b"x" * 20000)
    cfgp = tmp_path / "config.ini"
    settings.save(cfgp, {"MODELS_DIR": "models", "OUTPUT_DIR": "captures", "MODEL": "a.pt"})
    app = appmod.create_app(cfgp, tmp_path / "data", worker_factory=ErrorWorker,
                            names_loader=lambda p: {0: "car"}, device_detector=lambda: [],
                            grabber=lambda url: None)
    c = app.test_client()
    cid = _ready_cam(c)["id"]
    assert c.post(f"/api/capture/{cid}/start").status_code == 200
    done = threading.Event()
    threading.Thread(target=lambda: (c.get(f"/api/stream/{cid}").get_data(), done.set()), daemon=True).start()
    assert done.wait(3), "the MJPEG generator kept a server thread busy forever"


# ---------------------------------------------------------------- GPU lock covers the tensor -> CPU copies
def test_detector_copies_result_tensors_while_holding_the_gpu_lock(monkeypatch):
    import sys
    import types
    held = []

    class T:
        def __init__(self, v):
            self.v = v

        def cpu(self):
            held.append(gpu_lock._LOCK._is_owned())
            return self

        def numpy(self):
            return np.array(self.v, dtype=float)

        def int(self):
            return self

        def tolist(self):
            return list(self.v)

    class Boxes:
        xyxy, id, cls, conf = T([[1, 2, 30, 40]]), T([7]), T([0]), T([0.9])

    class FakeYOLO:
        names = {0: "car"}

        def __init__(self, *a, **k):
            pass

        def track(self, *a, **k):
            return [types.SimpleNamespace(boxes=Boxes())]

    mod = types.ModuleType("ultralytics")
    mod.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", mod)
    d = ce.YoloDetector("m.engine", "cuda:0", False, 0.3, 640, ["car"], [[[0, 0], [99, 0], [99, 99], [0, 99]]],
                        (100, 100))
    out = d(np.zeros((100, 100, 3), np.uint8))
    assert out and out[0].track_id == 7 and held and all(held)


# ---------------------------------------------------------------- worker shutdown / error paths
class _Cap:
    def __init__(self, n=50):
        self.n, self.i, self.released = n, 0, False

    def read(self):
        time.sleep(0.005)
        self.i += 1
        return (True, np.full((480, 640, 3), 60, np.uint8)) if self.i <= self.n else (False, None)

    def release(self):
        self.released = True


def test_worker_releases_the_stream_when_detection_crashes(tmp_path, monkeypatch):
    cap = _Cap()
    monkeypatch.setattr(camera_io, "open_stream", lambda *a, **k: camera_io.OpenResult(cap, "file"))
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"x")

    def boom(f):
        raise RuntimeError("detector exploded")

    w = ce.CameraWorker(_camera(tmp_path), cfg(OUTPUT_DIR=str(tmp_path / "o")), "cpu", source=str(vid),
                        classes=CLASSES, detector=boom)
    w.start()
    _wait_done(w)
    assert w.status()["state"] == "error" and cap.released


def test_a_buffered_vehicle_is_still_saved_when_the_worker_fails(tmp_path):
    vid = tmp_path / "v.mp4"
    _video(vid)
    out = tmp_path / "o"
    calls = {"n": 0}

    def flaky(f):
        calls["n"] += 1
        if calls["n"] > 3:
            raise RuntimeError("cuda error")
        return [det(1, (200, 200, 300, 300))]          # stays inside the ROI -> only buffered

    w = ce.CameraWorker(_camera(tmp_path), cfg(OUTPUT_DIR=str(out), WHEN="best"), "cpu", source=str(vid),
                        classes=CLASSES, detector=flaky)
    w.start()
    _wait_done(w)
    t = time.time()
    while time.time() - t < 3 and not list(out.rglob("*.jpg")):
        time.sleep(0.05)
    assert w.status()["state"] == "error"
    assert len(list(out.rglob("*.jpg"))) == 1 and w.status()["saved"] == 1


def test_reader_never_releases_the_capture_while_a_read_is_still_running():
    entered, unblock = threading.Event(), threading.Event()
    log = []

    class BlockingCap:
        def read(self):
            entered.set()
            unblock.wait(5)
            log.append("read returned")
            return False, None

        def release(self):
            log.append("release")

    rd = ce.LatestFrameReader(BlockingCap())
    rd.start()
    assert entered.wait(2)
    rd.close(timeout=0.1)                              # the read is stuck (e.g. a dead RTSP socket)
    assert "release" not in log
    unblock.set()
    t = time.time()
    while time.time() - t < 2 and "release" not in log:
        time.sleep(0.02)
    assert log[:2] == ["read returned", "release"]


def test_preview_jpeg_is_only_encoded_while_someone_watches(tmp_path, monkeypatch):
    encodes = []
    real = ce.cv2.imencode
    monkeypatch.setattr(ce.cv2, "imencode", lambda *a, **k: (encodes.append(1), real(*a, **k))[1])
    w = ce.CameraWorker(_camera(tmp_path), cfg(OUTPUT_DIR=str(tmp_path / "o")), "cpu", source="x",
                        classes=CLASSES, detector=lambda f: [])
    f = np.full((480, 640, 3), 60, np.uint8)
    w._preview(f, [])
    assert not encodes and w.latest_jpeg() is None      # nobody watching: no work
    w._preview(f, [])                                   # latest_jpeg() above = a viewer arrived
    assert encodes and w.latest_jpeg() is not None


def test_worker_status_is_consistent_under_concurrent_reads(tmp_path):
    vid = tmp_path / "v.mp4"
    _video(vid, n=60)
    w = ce.CameraWorker(_camera(tmp_path), cfg(OUTPUT_DIR=str(tmp_path / "o")), "cpu", source=str(vid),
                        classes=CLASSES, detector=lambda f: [det(1, (200, 200, 300, 300))])
    errors = []

    def poll():
        try:
            for _ in range(200):
                st = w.status()
                assert st["state"] in ("reconnecting", "connected", "stopping", "stopped")
                w.latest_jpeg()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    w.start()
    ts = [threading.Thread(target=poll) for _ in range(4)]
    [t.start() for t in ts]
    [t.join(10) for t in ts]
    _wait_done(w)
    w.stop()
    assert not errors and w.status()["saved"] == 1
