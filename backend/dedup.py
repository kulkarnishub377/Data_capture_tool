"""Duplicate suppression: track lock, IoU+time window, perceptual hash."""
import cv2
import numpy as np


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def phash(img: np.ndarray) -> int:
    """64-bit DCT perceptual hash."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(g)[:8, :8]
    bits = (low > np.median(low[1:].ravel() if low.size > 1 else low)).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class Deduper:
    """Suppress the SAME vehicle being saved twice (e.g. after a tracker ID switch).

    A saved vehicle is remembered while it keeps being detected (touch) and for `seconds`
    after it was last seen. A different vehicle arriving later at the same spot is NOT
    a duplicate.
    """

    def __init__(self, seconds=2, hash_dist=6, history=50, iou_thresh=0.5):
        self.seconds = seconds
        self.hash_dist = hash_dist
        self.history = history
        self.iou_thresh = iou_thresh
        self._tracks = set()
        self._recent = {}   # key -> [last_seen, box, hash]
        self._anon = 0
        self.skipped = 0

    def locked(self, track_id) -> bool:
        return track_id in self._tracks

    def lock(self, track_id) -> None:
        self._tracks.add(track_id)

    def touch(self, track_id, box, now) -> None:
        """A saved track is still being detected: keep it 'active' at its latest position."""
        e = self._recent.get(track_id)
        if e is not None:
            e[0], e[1] = now, tuple(box)

    def is_duplicate(self, box, img, now) -> bool:
        h = phash(img)
        for t, b, hh in self._recent.values():
            if now - t > self.seconds:
                continue
            if iou(box, b) > self.iou_thresh or _hamming(h, hh) <= self.hash_dist:
                self.skipped += 1
                return True
        return False

    def record(self, box, img, now, track_id=None) -> None:
        if track_id is None:
            track_id = ("anon", self._anon)
            self._anon += 1
        self._recent[track_id] = [now, tuple(box), phash(img)]
        while len(self._recent) > self.history:
            self._recent.pop(next(iter(self._recent)))
