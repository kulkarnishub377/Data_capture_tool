"""Per-camera capture: reader thread, detect+track loop, saver thread.

CaptureLogic holds all per-frame decisions (eligibility, once-per-track, dedup,
first/best) and is independent of threads, models and cameras.
"""
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import dedup
import roi
import storage

log = logging.getLogger("capture")

COCO_VEHICLES = ["car", "motorcycle", "bus", "truck"]
LOST_FRAMES = 10  # 'best' mode: finalize a track unseen for this many frames


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
    def __init__(self, cfg, poly, frame_size, classes):
        self.cfg = cfg
        self.poly = poly
        self.size = tuple(frame_size)
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
        if not roi.point_in_polygon(((x1 + x2) / 2, (y1 + y2) / 2), self.poly):
            return False
        return roi.box_overlap_ratio(d.box, self.poly, self.size) >= c["MIN_ROI_OVERLAP"]

    def _emit(self, track_id, cls, conf, box, frame, now, ts):
        """Lock the track, dedup on the vehicle crop; return a capture request or None."""
        self.dedup.lock(track_id)
        crop = storage._crop(frame, box, self.cfg["PAD_NORMAL"])
        if crop.size == 0:
            return None
        if self.dedup.is_duplicate(box, crop, now):
            return None
        self.dedup.record(box, crop, now)
        return {"track_id": track_id, "cls": cls, "conf": conf, "box": tuple(box),
                "frame": frame, "ts": ts}

    def process(self, frame, dets, now, ts) -> list:
        self._frame_no += 1
        out = []
        seen_ok = set()
        for d in dets:
            if self.dedup.locked(d.track_id) or not self.eligible(d):
                continue
            if self.cfg["WHEN"] == "first":
                r = self._emit(d.track_id, d.cls, d.conf, d.box, frame, now, ts)
                if r:
                    out.append(r)
            else:
                seen_ok.add(d.track_id)
                score = d.conf * (d.box[2] - d.box[0]) * (d.box[3] - d.box[1])
                cur = self._cand.get(d.track_id)
                if cur is None or score > cur["score"]:
                    self._cand[d.track_id] = {"score": score, "cls": d.cls, "conf": d.conf,
                                              "box": d.box, "frame": frame.copy(), "ts": ts,
                                              "seen": self._frame_no}
                else:
                    cur["seen"] = self._frame_no
        if self.cfg["WHEN"] == "best":
            for tid in list(self._cand):
                c = self._cand[tid]
                if tid not in seen_ok or self._frame_no - c["seen"] > LOST_FRAMES:
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
    """Runs a YOLO model with ByteTrack on the ROI bounding box of the frame."""

    def __init__(self, model_path, device, half, conf, imgsz, class_names, poly, frame_size):
        from ultralytics import YOLO
        self.model = YOLO(str(model_path))
        self.device = device
        self.half = bool(half and device.startswith("cuda"))
        self.conf, self.imgsz = conf, imgsz
        self.names = dict(self.model.names)
        self.class_ids = [i for i, n in self.names.items() if n in set(class_names)]
        self.bbox = roi.polygon_bbox(poly, frame_size)

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
        self._stop.set()
        for t in (self._thread, self._saver):
            if t and t.is_alive():
                t.join(timeout=10)
        self._set("stopped", "")

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

    def _open(self):
        if not self._is_file:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|analyzeduration;5000000|probesize;5000000|stimeout;5000000")
        cap = cv2.VideoCapture(str(self.source), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return cap

    def _build_detector(self, frame_size):
        if self.detector is None:
            self.detector = YoloDetector(
                Path(self.cfg["MODELS_DIR"]) / self.cfg["MODEL"], self.device, self.cfg["HALF"],
                self.cfg["CONF_THRESH"], self.cfg["IMG_SIZE"], self.classes,
                self.camera["roi"], frame_size)

    def _run(self):
        backoff = 1.0
        try:
            while not self._stop.is_set():
                cap = self._open()
                if cap is None:
                    self._set("reconnecting", "cannot open stream")
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
            return
        if self._logic:
            for r in self._logic.flush(time.time()):
                self._enqueue(r)
        self._drain()
        self._set("stopped", "")

    def _loop(self, cap) -> bool:
        """Process frames until the stream ends/drops. Returns True on clean EOF."""
        fails = 0
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                if self._is_file:
                    return True
                fails += 1
                if fails > 50:
                    return False
                time.sleep(0.02)
                continue
            fails = 0
            h, w = frame.shape[:2]
            if self._logic is None:
                self._build_detector((w, h))
                self._logic = CaptureLogic(self.cfg, self.camera["roi"], (w, h), self.classes)
            self._set("connected", "")
            dets = self.detector(frame)
            for r in self._logic.process(frame, dets, time.time(), datetime.now()):
                self._enqueue(r)
            self._preview(frame, dets)
        return False

    def _preview(self, frame, dets):
        img = frame.copy()
        cv2.polylines(img, [np.array(self.camera["roi"], np.int32)], True, (0, 255, 255), 2)
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
