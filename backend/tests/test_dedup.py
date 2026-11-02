import numpy as np
import dedup


def _img(seed):
    rng = np.random.default_rng(seed)
    return (rng.random((64, 64, 3)) * 255).astype("uint8")


def test_iou_identical_and_disjoint():
    assert dedup.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert dedup.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_track_lock():
    d = dedup.Deduper(seconds=30)
    assert not d.locked(7)
    d.lock(7)
    assert d.locked(7)


def test_overlapping_box_within_window_is_duplicate():
    d = dedup.Deduper(seconds=30)
    d.record((100, 100, 200, 200), _img(1), now=0.0)
    assert d.is_duplicate((105, 105, 205, 205), _img(2), now=10.0) is True
    assert d.skipped == 1


def test_overlapping_box_after_window_is_not_duplicate():
    d = dedup.Deduper(seconds=30)
    d.record((100, 100, 200, 200), _img(1), now=0.0)
    assert d.is_duplicate((105, 105, 205, 205), _img(2), now=31.0) is False


def test_same_image_different_box_is_duplicate_by_hash():
    d = dedup.Deduper(seconds=30)
    img = _img(3)
    d.record((0, 0, 50, 50), img, now=0.0)
    assert d.is_duplicate((400, 400, 450, 450), img.copy(), now=10.0) is True


def test_different_image_far_box_is_not_duplicate():
    d = dedup.Deduper(seconds=30)
    d.record((0, 0, 50, 50), _img(4), now=0.0)
    assert d.is_duplicate((400, 400, 450, 450), _img(5), now=100.0) is False
    assert d.skipped == 0


def test_same_image_after_window_is_not_duplicate():
    d = dedup.Deduper(seconds=30)
    img = _img(6)
    d.record((0, 0, 50, 50), img, now=0.0)
    assert d.is_duplicate((400, 400, 450, 450), img.copy(), now=500.0) is False


def test_different_car_same_spot_seconds_later_is_not_duplicate():
    d = dedup.Deduper(seconds=2)
    box = (100, 100, 200, 200)
    d.record(box, _img(10), now=0.0, track_id=1)
    d.touch(1, box, now=1.0)
    assert d.is_duplicate(box, _img(11), now=6.0) is False


def test_new_id_at_same_spot_while_saved_track_still_active_is_duplicate():
    d = dedup.Deduper(seconds=2)
    box = (100, 100, 200, 200)
    d.record(box, _img(12), now=0.0, track_id=1)
    d.touch(1, box, now=1.0)
    assert d.is_duplicate(box, _img(13), now=2.5) is True
