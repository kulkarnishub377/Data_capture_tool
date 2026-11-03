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
    s.set_rois(cam["id"], [[[0, 0], [10, 0], [10, 10]]], [640, 480])
    assert s.list()[0]["rois"] == [[[0, 0], [10, 0], [10, 10]]]
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
        s.set_rois(cam["id"], [[[0, 0], [1, 1]]], [640, 480])


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


def test_two_captures_same_millisecond_do_not_overwrite(tmp_path):
    a = _save(tmp_path, "frame")[0]
    b = _save(tmp_path, "frame")[0]
    assert a != b and a.exists() and b.exists()
    res = storage.list_captures(tmp_path)
    assert res["total"] == 2 and {i["class"] for i in res["items"]} == {"car"}


def test_class_with_underscore_parsed_with_collision_suffix(tmp_path):
    _save(tmp_path, "frame", cls="auto_rickshaw")
    _save(tmp_path, "frame", cls="auto_rickshaw")
    assert {i["class"] for i in storage.list_captures(tmp_path)["items"]} == {"auto_rickshaw"}


def test_mask_url_hides_password_even_with_at_sign():
    assert storage.mask_url("rtsp://admin:Arya@123@1.2.3.4:554/s") == "rtsp://admin:****@1.2.3.4:554/s"
    assert storage.mask_url("rtsp://1.2.3.4/s") == "rtsp://1.2.3.4/s"
    assert storage.mask_url("rtsp://admin:pw%40x@h:1/p") == "rtsp://admin:****@h:1/p"


def test_old_single_roi_is_migrated_to_rois(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps([{"id": "x", "name": "A", "url": "u", "enabled": True,
                              "roi": [[0, 0], [9, 0], [9, 9]], "frame_size": [640, 480]}]))
    cam = storage.CameraStore(p).list()[0]
    assert cam["rois"] == [[[0, 0], [9, 0], [9, 9]]] and "roi" not in cam


def test_set_rois_validates_every_polygon_and_allows_clearing(tmp_path):
    s = storage.CameraStore(tmp_path / "c.json")
    cam = s.add("A", "u")
    tri = [[0, 0], [9, 0], [9, 9]]
    with pytest.raises(ValueError):
        s.set_rois(cam["id"], [tri, [[0, 0], [1, 1]]], [640, 480])
    assert len(s.set_rois(cam["id"], [tri, tri], [640, 480])["rois"]) == 2
    assert s.set_rois(cam["id"], [], [640, 480])["rois"] == []


def _save_at(tmp_path, ts, cls="car", mode="both"):
    return storage.save_capture(tmp_path, "Cam-01", cls, ts, _frame(), (200, 150, 300, 250),
                                mode, 95, 0.15, 0.40, None)


def test_list_items_carry_a_readable_timestamp(tmp_path):
    _save(tmp_path, "frame")
    assert storage.list_captures(tmp_path)["items"][0]["ts"] == "2026-11-02 10:30:15"


def test_recent_captures_newest_first_one_item_per_vehicle_preferring_crop(tmp_path):
    _save_at(tmp_path, datetime(2026, 11, 2, 9, 0, 0, 1000))
    _save_at(tmp_path, datetime(2026, 11, 2, 9, 0, 5, 1000), cls="bus")
    _save_at(tmp_path, datetime(2026, 11, 1, 23, 0, 0, 1000))
    items = storage.recent_captures(tmp_path, limit=10)
    assert [i["class"] for i in items] == ["bus", "car", "car"]
    assert {i["kind"] for i in items} == {"crop"}
    assert [i["ts"] for i in items][0] == "2026-11-02 09:00:05"


def test_recent_captures_respects_limit_and_frame_only(tmp_path):
    for s in range(5):
        _save_at(tmp_path, datetime(2026, 11, 2, 9, 0, s, 1000), mode="frame")
    items = storage.recent_captures(tmp_path, limit=3)
    assert len(items) == 3 and {i["kind"] for i in items} == {"frame"}


@pytest.mark.parametrize("size, width", [((1080, 1920), 320), ((720, 1280), 320), ((100, 150), 320), ((1080, 1920), 1280)])
def test_thumbnail_is_made_in_memory_and_nothing_is_written(tmp_path, size, width):
    import cv2
    big = np.full((*size, 3), 90, dtype="uint8")
    p = storage.save_capture(tmp_path / "cap", "Cam-01", "car", TS, big, (10, 10, 50, 50), "frame",
                             95, 0.1, 0.4, None)[0]
    rel = p.relative_to(tmp_path / "cap").as_posix()
    before = sorted(x for x in tmp_path.rglob("*"))
    data = storage.make_thumbnail(tmp_path / "cap", rel, width)
    assert isinstance(data, bytes) and data[:2] == b"\xff\xd8"              # a JPEG
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == min(width, size[1])                                   # never upscaled
    assert abs(img.shape[0] / img.shape[1] - size[0] / size[1]) < 0.02          # aspect kept
    assert sorted(x for x in tmp_path.rglob("*")) == before                      # no file, no cache folder


def test_thumbnail_rejects_path_escape_and_missing_files(tmp_path):
    with pytest.raises(ValueError):
        storage.make_thumbnail(tmp_path, "../x.jpg", 200)
    with pytest.raises(FileNotFoundError):
        storage.make_thumbnail(tmp_path, "Cam-01/2026-11-02/frame/none.jpg", 200)


def test_list_captures_multi_class_and_kind_and_batch_delete(tmp_path):
    storage.save_capture(tmp_path, "Cam-01", "car", TS, _frame(), (10, 10, 50, 50), "frame", 95, 0.15, 0.4, None)
    storage.save_capture(tmp_path, "Cam-01", "bus", TS, _frame(), (10, 10, 50, 50), "crop", 95, 0.15, 0.4, None)
    storage.save_capture(tmp_path, "Cam-01", "truck", TS, _frame(), (10, 10, 50, 50), "frame", 95, 0.15, 0.4, None)

    # Multi-class
    res = storage.list_captures(tmp_path, cls="car,truck")
    assert res["total"] == 2
    assert {i["class"] for i in res["items"]} == {"car", "truck"}
    assert set(res["categories"]) == {"bus", "car", "truck"}

    # Kind filter
    crop_res = storage.list_captures(tmp_path, kind="crop")
    assert crop_res["total"] == 1 and crop_res["items"][0]["class"] == "bus"

    # Batch delete
    paths = [i["path"] for i in res["items"]]
    deleted, errors = storage.delete_captures_batch(tmp_path, paths)
    assert deleted == 2 and errors == []
    assert storage.list_captures(tmp_path)["total"] == 1


def test_list_captures_search_runs_over_all_pages_not_just_one(tmp_path):
    for i in range(5):
        storage.save_capture(tmp_path, "Cam-01", "car", datetime(2026, 11, 2, 10, 0, i), _frame(),
                             (10, 10, 50, 50), "frame", 95, 0.15, 0.4, None)
    storage.save_capture(tmp_path, "Gate-7", "auto_rickshaw", datetime(2026, 11, 1, 8, 0, 0), _frame(),
                         (10, 10, 50, 50), "frame", 95, 0.15, 0.4, None)
    res = storage.list_captures(tmp_path, q="RICKSHAW", per_page=2)       # the match is the oldest (last page)
    assert res["total"] == 1 and res["items"][0]["camera"] == "Gate-7"
    assert storage.list_captures(tmp_path, q="gate", per_page=2)["total"] == 1             # camera
    assert storage.list_captures(tmp_path, q="2026-11-02 10:00:03")["total"] == 1          # time text
    assert storage.list_captures(tmp_path, q="  ")["total"] == 6                           # blank = no filter


def test_batch_delete_reports_bad_entries_and_deletes_the_rest(tmp_path):
    p = _save(tmp_path, "frame")[0]
    rel = p.relative_to(tmp_path).as_posix()
    deleted, errors = storage.delete_captures_batch(tmp_path, [rel, "Cam-01/x/frame/missing.jpg", "../evil.jpg", 42])
    assert deleted == 1 and not p.exists()
    assert len(errors) == 3 and all(isinstance(e, str) for e in errors)


