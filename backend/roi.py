"""ROI polygon helpers."""
import cv2
import numpy as np


def point_in_polygon(pt, poly) -> bool:
    contour = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False) >= 0


def as_polygons(poly_or_polys):
    """Accept one polygon [[x,y],...] or a list of polygons; always return a list of polygons."""
    if poly_or_polys and poly_or_polys[0] and isinstance(poly_or_polys[0][0], (list, tuple)):
        return [list(p) for p in poly_or_polys]
    return [poly_or_polys]


def polygon_bbox(poly, frame_size):
    """Bounding box of one polygon or of the union of several."""
    w, h = frame_size
    pts = [pt for pg in as_polygons(poly) for pt in pg]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (max(0, int(min(xs))), max(0, int(min(ys))),
            min(w, int(max(xs))), min(h, int(max(ys))))


def expand_bbox(poly, frame_size, margin):
    """ROI bounding box grown by `margin` (fraction of its size) on every side, clamped to the frame."""
    w, h = frame_size
    x1, y1, x2, y2 = polygon_bbox(poly, frame_size)
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    return max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)


def make_mask(poly, frame_size):
    w, h = frame_size
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(pg, dtype=np.int32) for pg in as_polygons(poly)], 1)
    return mask


def box_overlap_ratio(box, poly, frame_size, mask=None) -> float:
    """Fraction of the box area that lies inside the polygon (pass a cached mask to skip rebuilding it)."""
    w, h = frame_size
    x1, y1, x2, y2 = [int(v) for v in box]
    x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    area = max(1, (x2 - x1) * (y2 - y1))
    if x2c <= x1c or y2c <= y1c:
        return 0.0
    if mask is None:
        mask = make_mask(poly, frame_size)
    return min(1.0, int(mask[y1c:y2c, x1c:x2c].sum()) / area)
