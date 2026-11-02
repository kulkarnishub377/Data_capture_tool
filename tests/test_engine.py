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


def mk(c=None):
    return ce.CaptureLogic(c or cfg(), POLY, SIZE, CLASSES)


def test_inside_track_saved_exactly_once():
    box = (200, 200, 300, 300)
    out = run(mk(), [(frame(), [det(1, box)]) for _ in range(30)])
    assert len(out) == 1 and out[0]["track_id"] == 1


def test_outside_track_never_saved():
    out = run(mk(), [(frame(), [det(1, (520, 200, 600, 300))]) for _ in range(20)])
    assert out == []


def test_mostly_outside_box_not_saved_even_if_centre_inside():
    # box straddles the ROI corner (100,100): centre (105,105) is inside, ~28% of the area is
    out = run(mk(), [(frame(), [det(1, (20, 20, 190, 190))]) for _ in range(5)])
    assert out == []


def test_track_id_switch_same_box_still_one_save():
    box = (200, 200, 300, 300)
    seq = [(frame(), [det(1, box)]) for _ in range(5)] + [(frame(), [det(2, box)]) for _ in range(5)]
    logic = mk()
    out = run(logic, seq)
    assert len(out) == 1 and logic.skipped >= 1


def test_class_not_selected_ignored():
    out = run(mk(), [(frame(), [det(1, (200, 200, 300, 300), cls="truck")]) for _ in range(5)])
    assert out == []


def test_low_conf_and_tiny_box_ignored():
    low = run(mk(), [(frame(), [det(1, (200, 200, 300, 300), conf=0.1)])])
    tiny = run(mk(), [(frame(), [det(1, (200, 200, 210, 210))])])
    assert low == [] and tiny == []


def test_two_distinct_vehicles_both_saved():
    f1, f2 = frame(), frame()
    cv2.rectangle(f1, (150, 150), (250, 250), (0, 0, 255), -1)
    cv2.circle(f2, (400, 300), 40, (255, 255, 0), -1)
    out = run(mk(), [(f1, [det(1, (150, 150, 250, 250))]), (f2, [det(2, (360, 260, 440, 340), cls="bus")])])
    assert len(out) == 2


def test_best_mode_saves_largest_confident_frame_once_when_track_leaves():
    c = cfg(WHEN="best")
    logic = mk(c)
    small, big = (200, 200, 260, 260), (200, 200, 320, 320)
    seq = [(frame(), [det(1, small, conf=0.5)]), (frame(), [det(1, big, conf=0.9)]),
           (frame(), [det(1, small, conf=0.6)]), (frame(), [])]
    out = run(logic, seq)
    assert len(out) == 1 and tuple(out[0]["box"]) == big


def test_best_mode_flush_saves_pending_track():
    logic = mk(cfg(WHEN="best"))
    run(logic, [(frame(), [det(1, (200, 200, 300, 300))])])
    assert len(logic.flush(now=5.0)) == 1


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
    return {"id": "c1", "name": name, "url": "", "roi": POLY, "frame_size": list(SIZE)}


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
