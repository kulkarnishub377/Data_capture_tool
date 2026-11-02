"""scripts/paddle_vehicle_probe.py helpers (PP-Vehicle expects raw 0-255 RGB, 640x640, plus scale_factor)."""
import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("paddle_vehicle_probe",
                                               Path(__file__).resolve().parents[2] / "scripts" / "paddle_vehicle_probe.py")
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)            # paddle itself is imported lazily, so this needs no Paddle


def test_prepare_image_is_rgb_chw_640_and_raw_0_to_255():
    frame = np.zeros((576, 704, 3), np.uint8)
    frame[:, :, 0] = 255                                    # pure BLUE in BGR
    x = probe.prepare_image(frame)
    assert x.shape == (1, 3, 640, 640) and x.dtype == np.float32
    assert x.max() == 255.0                                 # NOT divided by 255: the model normalises internally
    assert x[0, 2].min() == 255.0 and x[0, 0].max() == 0.0  # BGR -> RGB: blue ends up in channel 2


def test_scale_factor_is_height_then_width_ratio():
    f = np.zeros((576, 704, 3), np.uint8)
    assert np.allclose(probe.scale_factor(f), [[640 / 576, 640 / 704]])


def test_parse_detections_filters_by_confidence_sorts_and_ignores_padding_rows():
    rows = np.array([[0, 0.5, 10, 20, 110, 220],
                     [0, 0.9, 1, 2, 3, 4],
                     [0, 0.1, 5, 5, 6, 6],                  # below the threshold
                     [-1, 0.99, 0, 0, 0, 0]], np.float32)   # NMS padding row
    got = probe.parse_detections(rows, 0.3)
    assert [round(d["score"], 1) for d in got] == [0.9, 0.5]
    assert got[1]["box"] == (10, 20, 110, 220)


def test_parse_detections_handles_empty_output():
    assert probe.parse_detections(np.zeros((0, 6), np.float32), 0.3) == []
    assert probe.parse_detections(None, 0.3) == []
