import time
from datetime import datetime

import cv2
import numpy as np
import pytest

import capture_engine as ce
import settings

POLY = [[100, 100], [500, 100], [500, 400], [100, 400]]
SIZE = (640, 480)
CLASSES = ["car", "bus"]


def cfg(**kw):
    c = settings.load("nonexistent.ini")
    c.update(kw)
    return c


def frame(color=60):
    f = np.full((480, 640, 3), color, dtype="uint8")
    return f


def det(tid, box, cls="car", conf=0.9):
    return ce.Detection(tid, cls, conf, box)


def run(logic, frames_dets, start=0.0):
    out = []
    for i, (f, dets) in enumerate(frames_dets):
        out += logic.process(f, dets, now=start + i * 0.1, ts=datetime.now())
    return out


def test_default_classes_coco_vs_custom():
    coco = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 16: "dog"}
    assert ce.default_classes(coco) == ["car", "motorcycle", "bus", "truck"]
    custom = {0: "auto_rickshaw", 1: "bike", 2: "bus"}
    assert ce.default_classes(custom) == ["auto_rickshaw", "bike", "bus"]


def _video(path, n=30):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 480))
    for i in range(n):
        f = frame()
        cv2.rectangle(f, (200, 200), (300, 300), (0, 200, 0), -1)
        w.write(f)
    w.release()


def _camera(tmp_path, name="CamT"):
    return {"id": "c1", "name": name, "url": "", "rois": [POLY], "frame_size": list(SIZE)}


def _wait_done(w, timeout=15):
    t = time.time()
    while time.time() - t < timeout and w.status()["state"] not in ("stopped", "error"):
        time.sleep(0.05)


@pytest.mark.parametrize("meta", [False, True])
def test_worker_on_video_file_saves_one_image_and_optional_json(tmp_path, meta):
    vid = tmp_path / "v.mp4"
    _video(vid)
    out = tmp_path / "out"
    c = cfg(OUTPUT_DIR=str(out), SAVE_METADATA=meta)
    w = ce.CameraWorker(_camera(tmp_path), c, "cpu", source=str(vid), classes=CLASSES, model_name="stub",
                        detector=lambda f: [det(1, (200, 200, 300, 300))])
    w.start()
    _wait_done(w)
    w.stop()
    assert len(list(out.rglob("*.jpg"))) == 1
    assert len(list(out.rglob("*.json"))) == (1 if meta else 0)
    assert w.status()["saved"] == 1


def test_worker_bad_source_reconnects_without_crashing(tmp_path):
    w = ce.CameraWorker(_camera(tmp_path), cfg(OUTPUT_DIR=str(tmp_path / "o")), "cpu",
                        source=str(tmp_path / "nope.mp4"), classes=CLASSES, model_name="stub",
                        detector=lambda f: [])
    w.start()
    time.sleep(0.6)
    assert w.status()["state"] == "reconnecting"
    w.stop()
    assert w.status()["state"] == "stopped"


class _FakeCap:
    def __init__(self, n=60):
        self.i, self.n = 0, n

    def read(self):
        time.sleep(0.01)
        self.i += 1
        if self.i > self.n:
            time.sleep(0.2)
            return False, None
        return True, np.full((4, 4, 3), self.i, dtype="uint8")

    def release(self):
        pass


def test_latest_frame_reader_skips_ahead_for_slow_consumer():
    rd = ce.LatestFrameReader(_FakeCap(60))
    rd.start()
    time.sleep(0.3)
    got = rd.get(timeout=1.0)
    rd.stop()
    assert got is not None and int(got[0, 0, 0]) > 10  # newest frame, not frame #1


def test_worker_exception_sets_error_and_stops_threads(tmp_path):
    vid = tmp_path / "v.mp4"
    _video(vid)

    def boom(f):
        raise RuntimeError("cuda out of memory")

    w = ce.CameraWorker(_camera(tmp_path), cfg(OUTPUT_DIR=str(tmp_path / "o")), "cpu", source=str(vid),
                        classes=CLASSES, model_name="stub", detector=boom)
    w.start()
    _wait_done(w)
    time.sleep(0.6)
    st = w.status()
    assert st["state"] == "error" and "cuda" in st["message"]
    assert not w._saver.is_alive()


def test_run_with_timeout_returns_none_when_call_hangs():
    import app as appmod
    t0 = time.time()
    assert appmod.run_with_timeout(lambda: time.sleep(2) or "late", 0.2) is None
    assert time.time() - t0 < 1.0
    assert appmod.run_with_timeout(lambda: "ok", 1.0) == "ok"


def test_check_port_reports_unreachable_quickly():
    import app as appmod
    t0 = time.time()
    msg = appmod.check_port("10.255.255.1", 5549, timeout=0.5)  # non-routable address
    assert msg and "10.255.255.1:5549" in msg and time.time() - t0 < 3
    assert appmod.check_port("127.0.0.1", 1, timeout=0.5)       # closed local port
