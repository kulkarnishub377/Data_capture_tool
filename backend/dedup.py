"""Box / image similarity helpers used to recognise the same vehicle twice."""
import cv2
import numpy as np


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def same_object(a, b, iou_thresh=0.5, contain_thresh=0.8, min_area_ratio=0.4) -> bool:
    """True if two boxes in one frame are the same physical object boxed twice.

    Either they overlap strongly (IoU), or the smaller is mostly inside the larger AND is not tiny next to it.
    The size condition keeps a motorbike in front of a truck (fully inside the truck's box) a separate object.
    """
    if iou(a, b) > iou_thresh:
        return True
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    smaller, larger = min(area_a, area_b), max(area_a, area_b)
    if smaller <= 0:
        return False
    return inter / smaller >= contain_thresh and smaller / larger >= min_area_ratio


def phash(img: np.ndarray) -> int:
    """64-bit DCT perceptual hash (similar images -> small Hamming distance)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(g)[:8, :8]
    bits = (low > np.median(low[1:].ravel() if low.size > 1 else low)).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)
