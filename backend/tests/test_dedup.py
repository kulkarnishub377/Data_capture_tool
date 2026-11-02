import numpy as np

import dedup


def _img(seed):
    rng = np.random.default_rng(seed)
    return (rng.random((64, 64, 3)) * 255).astype("uint8")


def test_iou_identical_and_disjoint():
    assert dedup.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert dedup.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_same_object_boxed_twice_at_different_sizes_is_detected():
    # real case from a traffic video: one car boxed as 46x40 and 53x65 at the same spot (IoU only ~0.5)
    a = (338, 0, 384, 40)
    b = (334, -1, 387, 64)
    assert dedup.iou(a, b) < 0.65
    assert dedup.same_object(a, b) is True


def test_nearly_identical_boxes_are_the_same_object():
    assert dedup.same_object((100, 100, 200, 200), (105, 102, 205, 204)) is True


def test_small_vehicle_in_front_of_a_big_one_is_a_different_object():
    truck = (100, 100, 400, 400)
    motorbike = (200, 300, 240, 360)               # fully inside the truck's box, but tiny next to it
    assert dedup.same_object(truck, motorbike) is False


def test_side_by_side_vehicles_are_different_objects():
    assert dedup.same_object((100, 100, 200, 200), (210, 100, 310, 200)) is False
    assert dedup.same_object((0, 0, 10, 10), (50, 50, 60, 60)) is False


def test_degenerate_boxes_do_not_crash():
    assert dedup.same_object((5, 5, 5, 5), (5, 5, 9, 9)) is False


def test_phash_identical_images_match_and_different_images_do_not():
    a = _img(1)
    assert dedup.phash(a) == dedup.phash(a.copy())
    assert bin(dedup.phash(a) ^ dedup.phash(_img(2))).count("1") > 6
