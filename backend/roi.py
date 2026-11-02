"""ROI polygon helpers."""
import cv2
import numpy as np


def point_in_polygon(pt, poly) -> bool:
    contour = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False) >= 0


def polygon_bbox(poly, frame_size):
    w, h = frame_size
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (max(0, int(min(xs))), max(0, int(min(ys))),
            min(w, int(max(xs))), min(h, int(max(ys))))


def box_overlap_ratio(box, poly, frame_size) -> float:
    """Fraction of the box area that lies inside the polygon."""
    w, h = frame_size
    x1, y1, x2, y2 = [int(v) for v in box]
    x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    area = max(1, (x2 - x1) * (y2 - y1))
    if x2c <= x1c or y2c <= y1c:
        return 0.0
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(poly, dtype=np.int32)], 1)
    inside = int(mask[y1c:y2c, x1c:x2c].sum())
    return min(1.0, inside / area)
