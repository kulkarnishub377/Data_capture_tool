"""CaptureLogic: one capture per VEHICLE (not per tracker id), best frame buffered, older frames dropped."""
from datetime import datetime

import cv2
import numpy as np

import capture_logic as cl
import settings

POLY = [[100, 100], [500, 100], [500, 400], [100, 400]]     # the ROI
SECOND = [[560, 100], [620, 100], [620, 300], [560, 300]]
SIZE = (640, 480)
CLASSES = ["car", "bus", "truck"]
OUTSIDE = (520, 200, 560, 260)                              # a box whose centre is outside the ROI


def cfg(**kw):
    c = settings.load("nonexistent.ini")
    c.update(kw)
    return c


def frame(color=60):
    return np.full((480, 640, 3), color, dtype="uint8")


def det(tid, box, cls="car", conf=0.9):
    return cl.Detection(tid, cls, conf, box)


def mk(polys=None, **kw):
    return cl.CaptureLogic(cfg(**kw), polys or [POLY], SIZE, CLASSES)


def run(logic, seq, dt=0.1, start=0.0, flush=True):
    """Feed (frame, detections) pairs 0.1 s apart; returns every capture request, flushing at the end (= Stop)."""
    out, t = [], start
    for f, dets in seq:
        out += logic.process(f, dets, now=t, ts=datetime.now())
        t += dt
    if flush:
        out += logic.flush(now=t)
    return out


def box_at(cx, cy=250, w=80, h=80):
    return (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)


# ------------------------------------------------------------------ what qualifies
def test_vehicle_inside_roi_is_saved_exactly_once():
    out = run(mk(), [(frame(), [det(1, box_at(300))]) for _ in range(30)])
    assert len(out) == 1 and out[0]["track_id"] == 1


def test_vehicle_outside_roi_is_never_saved():
    assert run(mk(), [(frame(), [det(1, OUTSIDE)]) for _ in range(20)]) == []


def test_mostly_outside_box_is_not_saved_even_if_centre_is_inside():
    # straddles the ROI corner (100,100): centre inside, only ~28% of the area is
    assert run(mk(), [(frame(), [det(1, (20, 20, 190, 190))]) for _ in range(5)]) == []


def test_unselected_class_low_confidence_and_tiny_boxes_are_ignored():
    assert run(mk(), [(frame(), [det(1, box_at(300), cls="motorcycle")])] * 3) == []
    assert run(mk(), [(frame(), [det(1, box_at(300), conf=0.1)])] * 3) == []
    assert run(mk(), [(frame(), [det(1, (200, 200, 210, 210))])] * 3) == []


def test_vehicle_in_either_roi_is_saved_but_not_between_them():
    logic = mk(polys=[POLY, SECOND])
    seq = [(frame(), [det(1, (570, 150, 610, 250))]), (frame(), [det(2, box_at(300))]),
           (frame(), [det(3, (510, 150, 550, 250))])]            # centre x=530: in neither ROI
    assert sorted(r["track_id"] for r in run(logic, seq)) == [1, 2]


# ------------------------------------------------------------------ the buffer: keep best, drop older
def test_nothing_is_saved_while_the_vehicle_is_still_driving_through():
    logic = mk()
    out = run(logic, [(frame(), [det(1, box_at(300))]) for _ in range(20)], flush=False)
    assert out == []


def test_buffer_keeps_the_best_frame_and_drops_the_older_ones():
    f_small, f_big, f_small2 = frame(10), frame(20), frame(30)
    small, big = box_at(300, w=60, h=60), box_at(300, w=120, h=120)
    seq = [(f_small, [det(1, small, conf=0.5)]), (f_big, [det(1, big, conf=0.9)]),
           (f_small2, [det(1, small, conf=0.6)]), (frame(), [det(1, OUTSIDE)])]   # then it leaves the ROI
    out = run(mk(), seq)
    assert len(out) == 1
    assert out[0]["frame"] is f_big and tuple(out[0]["box"]) == big          # only the best frame survives


def test_a_better_later_frame_replaces_the_buffered_one():
    frames = [frame(i) for i in range(5)]
    seq = [(f, [det(1, box_at(300, w=60 + 15 * i, h=60 + 15 * i), conf=0.7)]) for i, f in enumerate(frames)]
    out = run(mk(), seq)
    assert len(out) == 1 and out[0]["frame"] is frames[-1]


def test_vehicle_is_saved_the_moment_it_leaves_the_roi():
    logic = mk()
    for i in range(5):
        assert logic.process(frame(), [det(1, box_at(300))], now=i * 0.1, ts=datetime.now()) == []
    out = logic.process(frame(), [det(1, OUTSIDE)], now=0.5, ts=datetime.now())
    assert len(out) == 1
    assert logic.process(frame(), [det(1, OUTSIDE)], now=0.6, ts=datetime.now()) == []   # and never again


def test_vehicle_that_stays_in_the_roi_is_saved_after_the_max_hold_time_only_once():
    logic = mk(MAX_HOLD_SECONDS=2)
    out = []
    for i in range(50):                                   # 5 s standing in the ROI
        out += logic.process(frame(), [det(1, box_at(300))], now=i * 0.1, ts=datetime.now())
    assert len(out) == 1


def test_vehicle_not_seen_for_longer_than_the_gap_is_saved():
    logic = mk(DEDUP_SECONDS=2)
    logic.process(frame(), [det(1, box_at(300))], now=0.0, ts=datetime.now())
    assert logic.process(frame(), [], now=1.5, ts=datetime.now()) == []        # might still come back
    assert len(logic.process(frame(), [], now=2.6, ts=datetime.now())) == 1    # gone for good


def test_detection_flicker_does_not_finalize_early():
    small, big = box_at(300, w=60, h=60), box_at(300, w=120, h=120)
    seq = [(frame(), [det(1, small, conf=0.5)]), (frame(), []), (frame(), []),
           (frame(), [det(1, big, conf=0.9)]), (frame(), [det(1, OUTSIDE)])]
    out = run(mk(), seq)
    assert len(out) == 1 and tuple(out[0]["box"]) == big


def test_prefers_a_fully_inside_view_over_a_bigger_partly_outside_one():
    f_partial, f_inside = frame(1), frame(2)
    partial = (40, 200, 190, 350)         # centre inside, ~60% of a 150px box inside the ROI
    inside = (200, 200, 310, 310)
    out = run(mk(), [(f_partial, [det(1, partial)]), (f_inside, [det(1, inside)])])
    assert out[0]["frame"] is f_inside


# ------------------------------------------------------------------ one vehicle = one capture (tracker ID switches)
def test_id_switch_while_buffered_is_still_one_vehicle():
    logic = mk()
    seq = [(frame(), [det(1, box_at(160 + 20 * i))]) for i in range(6)]              # id 1 drives right at 200 px/s
    seq += [(frame(), [])]                                                           # tracker loses it for a frame
    seq += [(frame(), [det(2, box_at(160 + 20 * i))]) for i in range(7, 13)]         # ...and re-labels it id 2
    seq += [(frame(), [det(2, OUTSIDE)])]
    out = run(logic, seq)
    assert len(out) == 1 and logic.skipped >= 1


def test_fast_vehicle_id_switch_is_linked_by_motion_prediction():
    logic = mk()
    seq = [(frame(), [det(1, box_at(130 + 60 * i))]) for i in range(4)]              # 600 px/s: x = 130..310
    seq += [(frame(), []), (frame(), [det(2, box_at(430))])]                         # new id exactly where it should be
    seq += [(frame(), [det(2, OUTSIDE)])]
    out = run(logic, seq)
    # the last box overlaps nothing (IoU 0, centre 120 px away), only prediction links them
    assert len(out) == 1


def test_id_switch_after_the_vehicle_was_already_saved_is_not_captured_again():
    logic = mk(WHEN="first")
    seq = [(frame(), [det(1, box_at(160 + 20 * i))]) for i in range(5)]
    seq += [(frame(), [det(2, box_at(160 + 20 * i))]) for i in range(5, 10)]
    out = run(logic, seq)
    assert len(out) == 1


def test_next_vehicle_in_the_same_lane_is_a_new_capture():
    f1, f2 = frame(1), frame(2)
    logic = mk()
    out = []
    for i in range(10):
        out += logic.process(f1, [det(1, box_at(300))], now=i * 0.1, ts=datetime.now())
    out += logic.process(f1, [det(1, OUTSIDE)], now=1.0, ts=datetime.now())             # car 1 leaves
    for i in range(10):
        out += logic.process(f2, [det(2, box_at(300))], now=5.0 + i * 0.1, ts=datetime.now())
    out += logic.flush(now=6.5)
    assert len(out) == 2 and out[0]["frame"] is f1 and out[1]["frame"] is f2


def test_two_cars_close_together_are_not_merged():
    seq = [(frame(), [det(1, box_at(250)), det(2, box_at(340))]) for _ in range(6)]
    assert len(run(mk(), seq)) == 2


def test_new_car_next_to_a_vehicle_that_is_still_visible_is_not_absorbed():
    logic = mk()
    seq = [(frame(), [det(1, box_at(250))]) for _ in range(5)]
    seq += [(frame(), [det(1, box_at(250)), det(2, box_at(330))]) for _ in range(5)]   # car 2 shows up beside car 1
    assert len(run(logic, seq)) == 2


def test_same_object_detected_as_two_classes_is_one_capture():
    seq = [(frame(), [det(1, box_at(300), cls="car", conf=0.9),
                      det(2, (262, 212, 342, 292), cls="truck", conf=0.6)]) for _ in range(6)]
    out = run(mk(), seq)
    assert len(out) == 1 and out[0]["cls"] == "car"


# ------------------------------------------------------------------ "first" mode and stopping
def test_first_mode_saves_immediately_once():
    logic = mk(WHEN="first")
    out = logic.process(frame(), [det(1, box_at(300))], now=0.0, ts=datetime.now())
    assert len(out) == 1
    assert logic.process(frame(), [det(1, box_at(300))], now=0.1, ts=datetime.now()) == []


def test_flush_saves_every_buffered_vehicle():
    logic = mk()
    logic.process(frame(), [det(1, box_at(200)), det(2, box_at(400))], now=0.0, ts=datetime.now())
    assert len(logic.flush(now=0.1)) == 2
    assert logic.flush(now=0.2) == []


def test_two_vehicles_with_different_looks_both_saved():
    f1, f2 = frame(), frame()
    cv2.rectangle(f1, (150, 150), (250, 250), (0, 0, 255), -1)
    cv2.circle(f2, (400, 300), 40, (255, 255, 0), -1)
    out = run(mk(), [(f1, [det(1, (150, 150, 250, 250))]), (f2, [det(2, (360, 260, 440, 340), cls="bus")])])
    assert len(out) == 2


# ------------------------------------------------------------------ real-world causes of duplicate captures
def test_one_object_boxed_twice_by_the_detector_is_one_capture():
    """Seen in a real traffic video: two tracker ids on the SAME car for many frames (boxes 46x40 and 53x65)."""
    seq = [(frame(), [det(3, (338, 200, 384, 240), conf=0.8), det(4, (334, 199, 387, 264), conf=0.7)])
           for _ in range(10)]
    seq += [(frame(), [det(4, OUTSIDE)])]
    out = run(mk(), seq)
    assert len(out) == 1


def test_alternating_double_boxes_never_split_into_two_vehicles():
    # the stronger box flips between the two ids from frame to frame
    seq = []
    for i in range(10):
        a, b = (338, 200, 384, 240), (334, 199, 387, 264)
        seq.append((frame(), [det(3, a, conf=0.8 if i % 2 else 0.6), det(4, b, conf=0.6 if i % 2 else 0.8)]))
    assert len(run(mk(), seq)) == 1


def test_low_frame_rate_every_detection_gets_a_new_id_but_it_is_one_vehicle():
    """CPU at ~2 fps: a car moves ~140 px between frames (IoU 0), so the tracker hands out a new id every frame."""
    logic = mk()
    out, t = [], 0.0
    for i in range(6):
        out += logic.process(frame(), [det(10 + i, box_at(130 + 70 * i, w=60, h=60))],
                             now=t, ts=datetime.now())
        t += 0.5
    out += logic.flush(now=t)
    assert len(out) == 1


def test_a_car_entering_behind_a_moving_one_is_not_absorbed_into_it():
    logic = mk()
    out = []
    for i in range(3):                                                      # car A heads down (y grows) at 160 px/s
        out += logic.process(frame(), [det(1, box_at(300, 150 + 80 * i, w=60, h=60))], now=i * 0.5, ts=datetime.now())
    # A is lost; car B appears BEHIND it (above its last position) 0.5 s later: must be a new vehicle
    out += logic.process(frame(), [det(2, box_at(300, 150, w=60, h=60))], now=1.5, ts=datetime.now())
    out += logic.flush(now=2.0)
    assert len(out) == 2


def test_dedup_seconds_zero_never_links_ids():
    logic = mk(DEDUP_SECONDS=0)
    seq = [(frame(), [det(1, box_at(300))]), (frame(), [det(2, box_at(300))])]
    assert len(run(logic, seq)) == 2
