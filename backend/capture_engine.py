"""Per-camera capture: latest-frame reader, detect+track loop, saver thread.

CaptureLogic holds all per-frame decisions (eligibility, once-per-track, dedup,
first/best) and is independent of threads, models and cameras.
"""
import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import camera_io
import dedup
import roi
import storage

log = logging.getLogger("capture")

COCO_VEHICLES = ["car", "motorcycle", "bus", "truck"]
LOST_FRAMES = 10        # 'best' mode: finalize a track unseen for this many frames
DETECT_MARGIN = 0.3     # detect on the ROI bounding box grown by this fraction, so a vehicle
                        # half outside the ROI is seen whole (not as a clipped slice)


@dataclass
class Detection:
    track_id: int
    cls: str
    conf: float
    box: tuple  # x1, y1, x2, y2 in full-frame pixels


def default_classes(names: dict) -> list:
    """COCO model -> vehicle classes; custom model -> all its classes."""
    vals = list(names.values())
    if "person" in vals and all(c in vals for c in COCO_VEHICLES):
        return list(COCO_VEHICLES)
    return vals


class CaptureLogic:
    """Decides, frame by frame, which detections become saved captures."""

    def __init__(self, cfg, polys, frame_size, classes):
        self.cfg = cfg
        self.polys = roi.as_polygons(polys)
        self.size = tuple(frame_size)
        self.mask = roi.make_mask(self.polys, self.size)   # built once, not per detection
        self.classes = set(classes)
        self.dedup = dedup.Deduper(seconds=cfg["DEDUP_SECONDS"])
        self._cand = {}  # best mode: track_id -> candidate dict
        self._frame_no = 0

    @property
    def skipped(self):
        return self.dedup.skipped

    def eligible(self, d: Detection) -> bool:
        c = self.cfg
        x1, y1, x2, y2 = d.box
        if d.cls not in self.classes or d.conf < c["CONF_THRESH"]:
            return False
        if min(x2 - x1, y2 - y1) < c["MIN_BOX"]:
            return False
        centre = ((x1 + x2) / 2, (y1 + y2) / 2)
        if not any(roi.point_in_polygon(centre, pg) for pg in self.polys):
            return False
        return roi.box_overlap_ratio(d.box, self.polys, self.size, mask=self.mask) >= c["MIN_ROI_OVERLAP"]

    def _emit(self, track_id, cls, conf, box, frame, now, ts):
        """Lock the track, dedup on the vehicle crop; return a capture request or None."""
        self.dedup.lock(track_id)
        crop = storage._crop(frame, box, self.cfg["PAD_NORMAL"])
        if crop.size == 0:
            return None
        if self.dedup.is_duplicate(box, crop, now):
            return None
        self.dedup.record(box, crop, now, track_id=track_id)
        return {"track_id": track_id, "cls": cls, "conf": conf, "box": tuple(box),
                "frame": frame, "ts": ts}

    def process(self, frame, dets, now, ts) -> list:
        self._frame_no += 1
        out, ok_ids, present = [], set(), set()
        best = self.cfg["WHEN"] == "best"
        for d in dets:
            present.add(d.track_id)
            if self.dedup.locked(d.track_id):
                self.dedup.touch(d.track_id, d.box, now)  # saved vehicle still in view
                continue
            if not self.eligible(d):
                continue
            if not best:
                r = self._emit(d.track_id, d.cls, d.conf, d.box, frame, now, ts)
                if r:
                    out.append(r)
                continue
            ok_ids.add(d.track_id)
            score = d.conf * (d.box[2] - d.box[0]) * (d.box[3] - d.box[1])
            cur = self._cand.get(d.track_id)
            if cur is None or score > cur["score"]:
                self._cand[d.track_id] = {"score": score, "cls": d.cls, "conf": d.conf, "box": d.box,
                                          "frame": frame.copy(), "ts": ts, "seen": self._frame_no}
            else:
                cur["seen"] = self._frame_no
        if best:
            for tid in list(self._cand):
                if tid in ok_ids:
                    continue
                c = self._cand[tid]
                left_roi = tid in present                              # seen, but no longer eligible
                lost = self._frame_no - c["seen"] > LOST_FRAMES        # not seen for a while
                if left_roi or lost:
                    del self._cand[tid]
                    r = self._emit(tid, c["cls"], c["conf"], c["box"], c["frame"], now, c["ts"])
                    if r:
                        out.append(r)
        return out

    def flush(self, now) -> list:
        out = []
        for tid, c in list(self._cand.items()):
            r = self._emit(tid, c["cls"], c["conf"], c["box"], c["frame"], now, c["ts"])
            if r:
                out.append(r)
        self._cand.clear()
        return out


class YoloDetector:
    """Runs a YOLO model with ByteTrack on the (margin-expanded) ROI bounding box."""

    def __init__(self, model_path, device, half, conf, imgsz, class_names, polys, frame_size):
        from ultralytics import YOLO
        self.model = YOLO(str(model_path))
        self.device = device
        self.half = bool(half and device.startswith("cuda"))
        self.conf, self.imgsz = conf, imgsz
        self.names = dict(self.model.names)
        self.class_ids = [i for i, n in self.names.items() if n in set(class_names)]
        self.bbox = roi.expand_bbox(polys, frame_size, DETECT_MARGIN)

    def __call__(self, frame):
        x1, y1, x2, y2 = self.bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or not self.class_ids:
            return []
        res = self.model.track(crop, persist=True, tracker="bytetrack.yaml", conf=self.conf,
                               imgsz=self.imgsz, device=self.device, half=self.half,
                               classes=self.class_ids, verbose=False)[0]
        boxes = res.boxes
        if boxes is None or boxes.id is None:
            return []
        out = []
        for b, tid, cls, cf in zip(boxes.xyxy.cpu().numpy(), boxes.id.int().cpu().tolist(),
                                   boxes.cls.int().cpu().tolist(), boxes.conf.cpu().tolist()):
            out.append(Detection(tid, self.names[cls], float(cf),
                                 (b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1)))
        return out


def load_model_names(model_path) -> dict:
    from ultralytics import YOLO
    return dict(YOLO(str(model_path)).names)


class LatestFrameReader:
    """Reads a stream as fast as it delivers and keeps only the newest frame.

    Slow inference then skips ahead to real time instead of falling behind the stream
    (which overflows FFmpeg's buffer and produces smeared/corrupt frames).
    """

    def __init__(self, cap):
        self._cap = cap
        self._cond = threading.Condition()
        self._frame, self._idx, self._seen = None, 0, 0
        self._stop = False
        self.failed = False
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        fails = 0
        while not self._stop:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                fails += 1
                if fails > 30:
                    with self._cond:
                        self.failed = True
                        self._cond.notify_all()
                    return
                time.sleep(0.02)
                continue
            fails = 0
            with self._cond:
                self._frame, self._idx = frame, self._idx + 1
                self._cond.notify_all()

    def get(self, timeout=1.0):
        """Newest frame not yet returned, or None on timeout / stream failure."""
        with self._cond:
            self._cond.wait_for(lambda: self._idx > self._seen or self.failed or self._stop, timeout)
            if self._idx > self._seen:
                self._seen = self._idx
                return self._frame
            return None

    def stop(self):
        self._stop = True
        with self._cond:
            self._cond.notify_all()
        if self._thread:
            self._thread.join(timeout=camera_io.READ_TIMEOUT_MS / 1000 + 1)


class CameraWorker:
    def __init__(self, camera, cfg, device, source=None, classes=None, model_name="",
                 detector=None):
        self.camera, self.cfg, self.device = camera, cfg, device
        self.source = source or camera["url"]
        self.classes = classes
        self.model_name = model_name
        self.detector = detector
        self.out_root = Path(cfg["OUTPUT_DIR"])
        self._stop = threading.Event()
        self._thread = None
        self._saver = None
        self._q = queue.Queue(maxsize=64)
        self._lock = threading.Lock()
        self._state, self._message = "stopped", ""
        self._saved = 0
        self._logic = None
        self._jpeg = None
        self._is_file = Path(str(self.source)).is_file()

    # ---- public -----------------------------------------------------------
    def start(self):
        self._stop.clear()
        self._set("reconnecting", "starting")
        self._saver = threading.Thread(target=self._save_loop, daemon=True)
        self._saver.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop and wait for the threads. Reports 'stopped' only once they have really exited."""
        self._stop.set()
        self._set("stopping", "")
        for t in (self._thread, self._saver):
            if t and t.is_alive():
                t.join(timeout=10)
        alive = any(t and t.is_alive() for t in (self._thread, self._saver))
        self._set("stopping" if alive else "stopped", "still shutting down" if alive else "")

    def status(self):
        with self._lock:
            return {"state": self._state, "message": self._message, "saved": self._saved,
                    "skipped": self._logic.skipped if self._logic else 0,
                    "device": self.device}

    def latest_jpeg(self):
        with self._lock:
            return self._jpeg

    # ---- internals --------------------------------------------------------
    def _set(self, state, message=""):
        with self._lock:
            self._state, self._message = state, message

    def _build_detector(self, frame_size):
        if self.detector is None:
            self.detector = YoloDetector(
                Path(self.cfg["MODELS_DIR"]) / self.cfg["MODEL"], self.device, self.cfg["HALF"],
                self.cfg["CONF_THRESH"], self.cfg["IMG_SIZE"], self.classes,
                self.camera["rois"], frame_size)

    def _unreachable(self):
        """Reason string when a network camera's port is closed, else None."""
        ep = camera_io.url_endpoint(self.source)
        return camera_io.check_port(ep[0], ep[1], timeout=3.0) if ep else None

    def _run(self):
        backoff = 1.0
        try:
            while not self._stop.is_set():
                problem = None if self._is_file else self._unreachable()
                cap = None if problem else camera_io.open_capture(self.source, self._is_file)
                if cap is None:
                    self._set("reconnecting", problem or "cannot open stream")
                    if self._stop.wait(backoff):
                        break
                    backoff = min(backoff * 2, 15)
                    continue
                backoff = 1.0
                ended = self._loop(cap)
                cap.release()
                if ended and self._is_file:
                    break
                if not self._stop.is_set():
                    self._set("reconnecting", "stream lost")
                    self._stop.wait(1.0)
        except Exception as e:  # noqa: BLE001 - surface in UI, never crash the app
            log.exception("worker failed")
            self._set("error", str(e))
            self._stop.set()    # lets the saver thread finish; Start will build a fresh worker
            return
        if self._logic:
            for r in self._logic.flush(time.time()):
                self._enqueue(r)
        self._drain()
        self._set("stopped", "")

    def _loop(self, cap) -> bool:
        """Process frames until the stream ends/drops. Returns True on clean end of a file."""
        reader = None if self._is_file else LatestFrameReader(cap)
        if reader:
            reader.start()
        try:
            while not self._stop.is_set():
                if reader:
                    frame = reader.get(1.0)
                    if frame is None:
                        if reader.failed:
                            return False
                        continue
                else:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        return True
                h, w = frame.shape[:2]
                if self._logic is None:
                    self._build_detector((w, h))
                    self._logic = CaptureLogic(self.cfg, self.camera["rois"], (w, h), self.classes)
                self._set("connected", "")
                dets = self.detector(frame)
                for r in self._logic.process(frame, dets, time.time(), datetime.now()):
                    self._enqueue(r)
                self._preview(frame, dets)
            return False
        finally:
            if reader:
                reader.stop()

    def _preview(self, frame, dets):
        img = frame.copy()
        for pg in roi.as_polygons(self.camera["rois"]):
            cv2.polylines(img, [np.array(pg, np.int32)], True, (0, 255, 255), 2)
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d.box]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{d.cls} {d.conf:.2f}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        if img.shape[1] > 960:
            s = 960 / img.shape[1]
            img = cv2.resize(img, (960, int(img.shape[0] * s)))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()

    def _enqueue(self, req):
        try:
            self._q.put_nowait(req)
        except queue.Full:
            log.warning("[%s] save queue full - capture dropped", self.camera["name"])

    def _drain(self):
        t = time.time()
        while not self._q.empty() and time.time() - t < 10:
            time.sleep(0.05)

    def _save_loop(self):
        c = self.cfg
        while True:
            producer_alive = self._thread is not None and self._thread.is_alive()
            if self._stop.is_set() and self._q.empty() and not producer_alive:
                break
            try:
                r = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            meta = None
            if c["SAVE_METADATA"]:
                meta = {"camera": self.camera["name"], "class": r["cls"], "conf": round(r["conf"], 4),
                        "track_id": r["track_id"], "bbox": [int(v) for v in r["box"]],
                        "timestamp": r["ts"].isoformat(), "model": self.model_name,
                        "save_mode": c["SAVE_MODE"]}
            try:
                storage.save_capture(self.out_root, self.camera["name"], r["cls"], r["ts"], r["frame"],
                                     r["box"], c["SAVE_MODE"], c["JPEG_QUALITY"], c["PAD_NORMAL"],
                                     c["PAD_BIKE"], meta)
                with self._lock:
                    self._saved += 1
            except Exception as e:  # noqa: BLE001
                log.error("save failed: %s", e)
                self._set(self._state, f"save failed: {e}")
