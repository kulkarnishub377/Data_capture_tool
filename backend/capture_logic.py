"""What gets saved: one capture per VEHICLE, chosen from a buffer of its frames.

Why this exists: a tracker gives a vehicle an ID, but IDs switch (occlusion, a missed detection, a fast vehicle
at a low frame rate) and then one vehicle looks like two. So this module keeps its own notion of a *vehicle*:

* A new tracker ID that appears where a recently-lost vehicle was expected (last box + its speed) is the SAME
  vehicle, not a new one. A vehicle that is still being detected never absorbs another track.
* Frames are buffered per vehicle: only the best one is kept, a better frame replaces (drops) the older one.
* The vehicle is saved once, when it leaves the ROI, when it has not been seen for `DEDUP_SECONDS`, after
  `MAX_HOLD_SECONDS` of standing in the ROI, or when capture stops (flush).
"""
import math
from dataclasses import dataclass, field

import dedup
import roi


@dataclass
class Detection:
    track_id: int
    cls: str
    conf: float
    box: tuple  # x1, y1, x2, y2 in full-frame pixels


@dataclass
class _Vehicle:
    vid: int
    cx: float = 0.0
    cy: float = 0.0
    w: float = 0.0
    h: float = 0.0
    vx: float = 0.0              # px / second, smoothed
    vy: float = 0.0
    last_seen: float = 0.0
    n_seen: int = 0
    first_eligible: float = None
    best: dict = None            # the buffered capture candidate (None once saved / if never eligible)
    saved: bool = False
    tracks: set = field(default_factory=set)


def _centre(box):
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


class CaptureLogic:
    CLEANUP_AFTER = 5.0          # seconds beyond the gap before an idle vehicle is forgotten

    def __init__(self, cfg, polys, frame_size, classes):
        self.cfg = cfg
        self.polys = roi.as_polygons(polys)
        self.size = tuple(frame_size)
        self.mask = roi.make_mask(self.polys, self.size)       # built once, not per detection
        self.classes = set(classes)
        self._vehicles = {}      # vid -> _Vehicle
        self._by_track = {}      # tracker id -> vid
        self._next_vid = 1
        self.merged = 0          # tracker-id switches linked back to the same vehicle
        self.suppressed = 0      # double detections of one object dropped

    @property
    def skipped(self):
        """Duplicates that were avoided (shown in the UI)."""
        return self.merged + self.suppressed

    # ---------------------------------------------------------------- per-detection assessment
    def _assess(self, d: Detection):
        """-> (eligible, centre_inside_roi, overlap)."""
        c = self.cfg
        x1, y1, x2, y2 = d.box
        cx, cy = _centre(d.box)
        inside = any(roi.point_in_polygon((cx, cy), pg) for pg in self.polys)
        if not inside:
            return False, False, 0.0
        overlap = roi.box_overlap_ratio(d.box, self.polys, self.size, mask=self.mask)
        ok = (d.cls in self.classes and d.conf >= c["CONF_THRESH"] and min(x2 - x1, y2 - y1) >= c["MIN_BOX"]
              and overlap >= c["MIN_ROI_OVERLAP"])
        return ok, True, overlap

    def _score(self, d: Detection, overlap: float) -> float:
        """Bigger, more confident, fully inside the ROI and not cut by the frame edge is better."""
        x1, y1, x2, y2 = d.box
        w, h = self.size
        cut = 0.6 if (x1 <= 1 or y1 <= 1 or x2 >= w - 1 or y2 >= h - 1) else 1.0
        return d.conf * (x2 - x1) * (y2 - y1) * overlap ** 2 * cut

    def _suppress_overlaps(self, dets):
        kept = []
        for d in sorted(dets, key=lambda d: -d.conf):
            if any(dedup.same_object(d.box, k.box) for k in kept):         # one object boxed twice
                self.suppressed += 1
                continue
            kept.append(d)
        return kept

    # ---------------------------------------------------------------- vehicle identity
    def _new_vehicle(self):
        v = _Vehicle(self._next_vid)
        self._next_vid += 1
        self._vehicles[v.vid] = v
        return v

    def _find_lost_vehicle(self, box, now, matched):
        """A vehicle that is NOT detected in this frame, was seen within the gap, and is expected near `box`.

        Once its speed is known the expected spot is last position + speed x time, with a tolerance that grows
        with the time since it was seen. With a single sighting it could have gone any way, so the tolerance is
        wider (a CPU at ~2 fps moves a car several of its own lengths between frames)."""
        gap = float(self.cfg["DEDUP_SECONDS"])
        cx, cy = _centre(box)
        size = max(box[2] - box[0], box[3] - box[1])
        best, best_d = None, None
        for v in self._vehicles.values():
            if v.vid in matched:
                continue
            dt = now - v.last_seen
            if dt > gap:
                continue
            s = max(size, v.w, v.h)
            d_last = math.hypot(cx - v.cx, cy - v.cy)
            if v.n_seen >= 2:
                d = math.hypot(cx - (v.cx + v.vx * dt), cy - (v.cy + v.vy * dt))
                ok = d <= s * (1 + 1.5 * dt) or d_last <= s          # on its predicted path, or it simply stopped
                d = min(d, d_last)
            else:
                d = d_last
                ok = d <= s * (1 + 3 * dt)
            if ok and (best is None or d < best_d):
                best, best_d = v, d
        return best

    @staticmethod
    def _update_motion(v: _Vehicle, box, now):
        cx, cy = _centre(box)
        dt = now - v.last_seen
        if v.n_seen >= 1 and dt > 1e-3:
            vx, vy = (cx - v.cx) / dt, (cy - v.cy) / dt
            v.vx = vx if v.n_seen == 1 else 0.5 * v.vx + 0.5 * vx
            v.vy = vy if v.n_seen == 1 else 0.5 * v.vy + 0.5 * vy
        v.cx, v.cy = cx, cy
        v.w, v.h = box[2] - box[0], box[3] - box[1]
        v.last_seen = now
        v.n_seen += 1

    # ---------------------------------------------------------------- saving
    @staticmethod
    def _request(v: _Vehicle):
        b = v.best
        return {"track_id": b["track_id"], "vehicle_id": v.vid, "cls": b["cls"], "conf": b["conf"],
                "box": tuple(b["box"]), "frame": b["frame"], "ts": b["ts"]}

    def _finalize(self, v: _Vehicle):
        """Save the buffered best frame once; the buffer (and its frame) is released."""
        req = self._request(v)
        v.saved, v.best = True, None
        return req

    # ---------------------------------------------------------------- main entry
    def process(self, frame, dets, now, ts) -> list:
        out = []
        gap = float(self.cfg["DEDUP_SECONDS"])
        first_mode = self.cfg["WHEN"] == "first"
        max_hold = float(self.cfg["MAX_HOLD_SECONDS"])

        matched = {}   # vid -> detection of this frame
        fresh = []     # detections whose tracker id has not been seen before
        for d in self._suppress_overlaps(dets):
            vid = self._by_track.get(d.track_id)
            if vid is None:
                fresh.append(d)
            elif vid not in matched:
                matched[vid] = d
        # an unknown tracker id: the same vehicle that was just lost, or a genuinely new one
        for d in sorted(fresh, key=lambda d: -d.conf):
            v = self._find_lost_vehicle(d.box, now, matched)
            if v is None:
                v = self._new_vehicle()
            else:
                self.merged += 1
            v.tracks.add(d.track_id)
            self._by_track[d.track_id] = v.vid
            matched[v.vid] = d

        for vid, d in matched.items():
            v = self._vehicles[vid]
            self._update_motion(v, d.box, now)
            if v.saved:
                continue                                    # saved already: still absorbs its own ID switches
            ok, inside, overlap = self._assess(d)
            if ok:
                if v.first_eligible is None:
                    v.first_eligible = now
                if first_mode:
                    v.best = {"track_id": d.track_id, "cls": d.cls, "conf": d.conf, "box": d.box,
                              "frame": frame, "ts": ts}
                    out.append(self._finalize(v))
                    continue
                score = self._score(d, overlap)
                if v.best is None or score > v.best["score"]:       # better frame replaces the buffered one
                    v.best = {"score": score, "track_id": d.track_id, "cls": d.cls, "conf": d.conf,
                              "box": d.box, "frame": frame, "ts": ts}
                if now - v.first_eligible >= max_hold:
                    out.append(self._finalize(v))
            elif v.best is not None and not inside:                  # it drove out of the ROI: done
                out.append(self._finalize(v))

        for v in list(self._vehicles.values()):
            if v.vid in matched:
                continue
            unseen = now - v.last_seen
            if not v.saved and v.best is not None and unseen > gap:  # gone for good
                out.append(self._finalize(v))
            if v.best is None and unseen > gap + self.CLEANUP_AFTER:
                self._forget(v)
        return out

    def _forget(self, v: _Vehicle):
        self._vehicles.pop(v.vid, None)
        for t in v.tracks:
            self._by_track.pop(t, None)

    def flush(self, now) -> list:
        """Capture is stopping: save every vehicle that is still buffered."""
        return [self._finalize(v) for v in list(self._vehicles.values()) if not v.saved and v.best is not None]
