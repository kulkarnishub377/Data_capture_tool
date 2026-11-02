import json
from datetime import datetime

import numpy as np
import pytest

import storage

TS = datetime(2026, 11, 2, 10, 30, 15, 123000)


def _frame():
    return np.full((480, 640, 3), 120, dtype="uint8")


def _save(root, mode, cls="car", meta=None):
    return storage.save_capture(root, "Cam-01", cls, TS, _frame(), (200, 150, 300, 250),
                                mode, 95, 0.15, 0.40, meta)


def test_camera_store_crud_and_roi(tmp_path):
    s = storage.CameraStore(tmp_path / "cameras.json")
    cam = s.add("Cam-01", "rtsp://x", device="cuda:1")
    assert s.list()[0]["name"] == "Cam-01"
    s.set_roi(cam["id"], [[0, 0], [10, 0], [10, 10]], [640, 480])
    assert s.list()[0]["roi"] == [[0, 0], [10, 0], [10, 10]]
    s.update(cam["id"], enabled=False)
    assert s.list()[0]["enabled"] is False
    s.delete(cam["id"])
    assert s.list() == []


def test_camera_store_write_is_atomic(tmp_path, monkeypatch):
    s = storage.CameraStore(tmp_path / "cameras.json")
    s.add("A", "rtsp://a")
    before = (tmp_path / "cameras.json").read_text()
    monkeypatch.setattr(storage.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        s.add("B", "rtsp://b")
    assert (tmp_path / "cameras.json").read_text() == before


def test_set_roi_requires_three_points(tmp_path):
    s = storage.CameraStore(tmp_path / "c.json")
    cam = s.add("A", "rtsp://a")
    with pytest.raises(ValueError):
        s.set_roi(cam["id"], [[0, 0], [1, 1]], [640, 480])


@pytest.mark.parametrize("mode,frames,crops", [("frame", 1, 0), ("crop", 0, 1), ("both", 1, 1)])
def test_save_modes(tmp_path, mode, frames, crops):
    paths = _save(tmp_path, mode)
    day = tmp_path / "Cam-01" / "2026-11-02"
    assert len(list((day / "frame").glob("*.jpg"))) == frames if (day / "frame").exists() else frames == 0
    assert len(list((day / "crop").glob("*.jpg"))) == crops if (day / "crop").exists() else crops == 0
    assert len(paths) == frames + crops
    assert paths[0].name == "Cam-01_car_20261102_103015_123.jpg"


def test_metadata_off_writes_no_json(tmp_path):
    _save(tmp_path, "frame", meta=None)
    assert list(tmp_path.rglob("*.json")) == []


def test_metadata_on_writes_json_beside_image(tmp_path):
    paths = _save(tmp_path, "frame", meta={"class": "car", "conf": 0.9})
    j = paths[0].with_suffix(".json")
    assert json.loads(j.read_text())["class"] == "car"


def test_bike_crop_uses_larger_padding(tmp_path):
    import cv2
    car = _save(tmp_path, "crop", cls="car")[0]
    bike = storage.save_capture(tmp_path, "Cam-01", "bike", TS, _frame(), (200, 150, 300, 250),
                                "crop", 95, 0.15, 0.40, None)[0]
    ch, bh = cv2.imread(str(car)).shape[0], cv2.imread(str(bike)).shape[0]
    assert bh > ch


def test_list_captures_filters_by_class(tmp_path):
    _save(tmp_path, "frame", cls="car")
    storage.save_capture(tmp_path, "Cam-01", "bus", datetime(2026, 11, 2, 11, 0, 0, 1000),
                         _frame(), (10, 10, 100, 100), "frame", 95, 0.15, 0.4, None)
    res = storage.list_captures(tmp_path, cls="bus")
    assert res["total"] == 1 and res["items"][0]["class"] == "bus"
    assert storage.list_captures(tmp_path)["total"] == 2


@pytest.mark.parametrize("bad", ["../x.jpg", "/etc/passwd", "C:/Windows/x.jpg", "Cam-01\\..\\x.jpg", "Cam-01/../../x"])
def test_safe_capture_path_rejects_escape(tmp_path, bad):
    with pytest.raises(ValueError):
        storage.safe_capture_path(tmp_path, bad)


def test_delete_capture_removes_json_too(tmp_path):
    paths = _save(tmp_path, "frame", meta={"a": 1})
    rel = paths[0].relative_to(tmp_path).as_posix()
    storage.delete_capture(tmp_path, rel)
    assert not paths[0].exists() and not paths[0].with_suffix(".json").exists()
