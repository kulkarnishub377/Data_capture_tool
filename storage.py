"""JSON camera store, capture file writing and gallery scanning. No database."""
import json
import os
import re
import threading
import uuid
from pathlib import Path

import cv2

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(name: str) -> str:
    return _SAFE.sub("_", str(name)).strip("._") or "cam"


class CameraStore:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.RLock()

    def _read(self):
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    def _write(self, cams):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cams, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def list(self):
        with self._lock:
            return self._read()

    def get(self, cam_id):
        return next((c for c in self.list() if c["id"] == cam_id), None)

    def add(self, name, url, device=None):
        with self._lock:
            cams = self._read()
            cam = {"id": uuid.uuid4().hex[:8], "name": name, "url": url, "enabled": True,
                   "roi": [], "frame_size": None, "device": device}
            cams.append(cam)
            self._write(cams)
            return cam

    def update(self, cam_id, **fields):
        with self._lock:
            cams = self._read()
            for c in cams:
                if c["id"] == cam_id:
                    c.update({k: v for k, v in fields.items() if k != "id"})
                    self._write(cams)
                    return c
            raise KeyError(cam_id)

    def delete(self, cam_id):
        with self._lock:
            self._write([c for c in self._read() if c["id"] != cam_id])

    def set_roi(self, cam_id, points, frame_size):
        if len(points) < 3:
            raise ValueError("ROI needs at least 3 points")
        return self.update(cam_id, roi=[[int(x), int(y)] for x, y in points],
                           frame_size=list(frame_size))


def _crop(frame, box, pad):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    return frame[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]


def save_capture(root, cam, cls, ts, frame, box, mode, quality, pad_normal, pad_bike, meta):
    """Write the image(s); a same-named .json only when meta is not None."""
    day = Path(root) / safe_name(cam) / ts.strftime("%Y-%m-%d")
    stem = f"{safe_name(cam)}_{safe_name(cls)}_{ts.strftime('%Y%m%d_%H%M%S')}_{ts.microsecond // 1000:03d}"
    pad = pad_bike if cls in ("bike", "motorcycle", "bicycle") else pad_normal
    targets = []
    if mode in ("frame", "both"):
        targets.append(("frame", frame))
    if mode in ("crop", "both"):
        targets.append(("crop", _crop(frame, box, pad)))
    written = []
    for sub, img in targets:
        if img is None or img.size == 0:
            continue
        d = day / sub
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{stem}.jpg"
        if not cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]):
            raise OSError(f"could not write {p}")
        if meta is not None:
            p.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        written.append(p)
    return written


def _class_from_name(cam_dir_name, stem):
    rest = stem[len(cam_dir_name) + 1:] if stem.startswith(cam_dir_name + "_") else stem
    parts = rest.rsplit("_", 3)
    return parts[0] if len(parts) == 4 else rest


def list_captures(root, camera=None, date=None, cls=None, page=1, per_page=60):
    root = Path(root)
    items = []
    if root.exists():
        for p in root.rglob("*.jpg"):
            rel = p.relative_to(root)
            if len(rel.parts) != 4:
                continue
            cam_dir, day, kind, _ = rel.parts
            c = _class_from_name(cam_dir, p.stem)
            if camera and cam_dir != safe_name(camera):
                continue
            if date and day != date:
                continue
            if cls and c != cls:
                continue
            items.append({"path": rel.as_posix(), "camera": cam_dir, "date": day,
                          "kind": kind, "class": c, "name": p.name})
    items.sort(key=lambda i: i["name"].rsplit("_", 3)[-3:], reverse=True)
    total = len(items)
    start = (max(1, page) - 1) * per_page
    return {"total": total, "page": page, "per_page": per_page,
            "items": items[start:start + per_page]}


def safe_capture_path(root, rel) -> Path:
    root = Path(root).resolve()
    if not rel or "\\" in rel or rel.startswith("/") or re.match(r"^[A-Za-z]:", rel):
        raise ValueError("invalid path")
    p = (root / rel).resolve()
    if root != p and root not in p.parents:
        raise ValueError("path escapes capture root")
    return p


def delete_capture(root, rel):
    p = safe_capture_path(root, rel)
    if p.suffix.lower() != ".jpg" or not p.is_file():
        raise FileNotFoundError(rel)
    p.unlink()
    j = p.with_suffix(".json")
    if j.exists():
        j.unlink()
