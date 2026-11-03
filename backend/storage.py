"""JSON camera store, capture file writing and gallery scanning. No database."""
import json
import os
import re
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np

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
            cams = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        for c in cams:  # migrate the old single "roi" to the "rois" list
            old = c.pop("roi", None)
            if "rois" not in c:
                c["rois"] = [old] if old and len(old) >= 3 else []
        return cams

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
                   "rois": [], "frame_size": None, "device": device}
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

    def set_rois(self, cam_id, rois, frame_size):
        """Replace all ROI polygons of a camera. Every polygon needs >= 3 points; [] clears them."""
        clean = []
        for poly in rois:
            if len(poly) < 3:
                raise ValueError("every ROI needs at least 3 points")
            clean.append([[int(x), int(y)] for x, y in poly])
        return self.update(cam_id, rois=clean, frame_size=list(frame_size))


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
    n = 1
    base = stem
    while any((day / sub / f"{stem}.jpg").exists() for sub, _ in targets):
        n += 1
        stem = f"{base}_{n}"
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


_NAME_RE = re.compile(r"^(?P<rest>.+)_(?P<d>\d{8})_(?P<t>\d{6})_(?P<ms>\d{3})(?:_(?P<n>\d+))?$")


def _parse_name(cam_dir_name, stem):
    """-> (class, sort_key). Class names may contain underscores (auto_rickshaw)."""
    m = _NAME_RE.match(stem)
    if not m:
        return stem, ("", "", "", 0)
    rest = m["rest"]
    if rest.startswith(cam_dir_name + "_"):
        rest = rest[len(cam_dir_name) + 1:]
    return rest, (m["d"], m["t"], m["ms"], int(m["n"] or 1))


def _ts_text(key):
    d, t = key[0], key[1]
    return f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}:{t[4:6]}" if d and t else ""


def list_captures(root, camera=None, date=None, cls=None, kind=None, sort="newest", page=1, per_page=60, q=None):
    """Filtered, sorted page of captures, plus every class / camera folder / date found (for the filter controls).

    `q` searches (case-insensitive) the file name, camera, class and time text across ALL pages."""
    root = Path(root)
    needle = (q or "").strip().lower()
    items = []
    all_categories = set()
    all_cameras = set()
    all_dates = set()

    classes = None
    if cls:
        if isinstance(cls, (list, set, tuple)):
            classes = {str(x).strip() for x in cls if str(x).strip()}
        else:
            classes = {x.strip() for x in str(cls).split(",") if x.strip()}

    if root.exists():
        for p in root.rglob("*.jpg"):
            rel = p.relative_to(root)
            if len(rel.parts) != 4:
                continue
            cam_dir, day, k, _ = rel.parts
            c, key = _parse_name(cam_dir, p.stem)
            all_categories.add(c)
            all_cameras.add(cam_dir)
            all_dates.add(day)

            if camera and cam_dir != safe_name(camera):
                continue
            if date and day != date:
                continue
            if classes and c not in classes:
                continue
            if kind and kind != "all" and k != kind:
                continue
            ts = _ts_text(key)
            if needle and not any(needle in s.lower() for s in (p.name, cam_dir, c, ts)):
                continue
            items.append({"path": rel.as_posix(), "camera": cam_dir, "date": day,
                          "kind": k, "class": c, "name": p.name, "ts": ts, "_key": key})
    items.sort(key=lambda i: i["_key"], reverse=(sort != "oldest"))
    for i in items:
        del i["_key"]
    total = len(items)
    start = (max(1, page) - 1) * per_page
    return {"total": total, "page": page, "per_page": per_page,
            "items": items[start:start + per_page],
            "categories": sorted(all_categories),
            "cameras": sorted(all_cameras),
            "dates": sorted(all_dates, reverse=True)}


def delete_captures_batch(root, rel_paths):
    """Delete several captures; one bad entry never stops the rest. Returns (deleted_count, [error text, ...])."""
    deleted, errors = 0, []
    for rel in rel_paths:
        try:
            if not isinstance(rel, str):
                raise ValueError("not a path")
            delete_capture(root, rel)
            deleted += 1
        except FileNotFoundError:
            errors.append(f"{rel}: not found")
        except (ValueError, OSError) as e:
            errors.append(f"{rel}: {e}")
    return deleted, errors


def safe_capture_path(root, rel) -> Path:
    root = Path(root).resolve()
    if not rel or "\\" in rel or rel.startswith("/") or re.match(r"^[A-Za-z]:", rel):
        raise ValueError("invalid path")
    p = (root / rel).resolve()
    if root != p and root not in p.parents:
        raise ValueError("path escapes capture root")
    return p


def delete_capture(root, rel):
    """Delete a capture and its .json."""
    p = safe_capture_path(root, rel)
    if p.suffix.lower() != ".jpg" or not p.is_file():
        raise FileNotFoundError(rel)
    p.unlink()
    j = p.with_suffix(".json")
    if j.exists():
        j.unlink()


def mask_url(url: str) -> str:
    """Hide the password in scheme://user:password@host/... (password may itself contain '@')."""
    m = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*://)([^/]*)(.*)$", url or "")
    if not m:
        return url
    scheme, auth, rest = m.groups()
    userinfo, at, host = auth.rpartition("@")
    if not at or ":" not in userinfo:
        return url
    return f"{scheme}{userinfo.split(':', 1)[0]}:****@{host}{rest}"


def recent_captures(root, limit=24):
    """Newest captures across all cameras, one item per vehicle (the crop if both crop and frame exist).

    Only the newest date folders are read, so this stays fast with tens of thousands of files.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    items = []
    for cam in root.iterdir():
        if not cam.is_dir():
            continue
        got = 0
        for day in sorted((d for d in cam.iterdir() if d.is_dir()), reverse=True):
            by_stem = {}
            for kind in ("frame", "crop"):          # crop is written last so it wins
                kd = day / kind
                if kd.is_dir():
                    for p in kd.glob("*.jpg"):
                        by_stem[p.stem] = (p, kind)
            for stem, (p, kind) in by_stem.items():
                c, key = _parse_name(cam.name, stem)
                items.append({"path": p.relative_to(root).as_posix(), "camera": cam.name, "date": day.name,
                              "kind": kind, "class": c, "name": p.name, "ts": _ts_text(key), "_key": key})
            got += len(by_stem)
            if got >= limit:
                break
    items.sort(key=lambda i: i["_key"], reverse=True)
    items = items[:limit]
    for i in items:
        del i["_key"]
    return items


_REDUCED = ((8, cv2.IMREAD_REDUCED_COLOR_8), (4, cv2.IMREAD_REDUCED_COLOR_4), (2, cv2.IMREAD_REDUCED_COLOR_2))
_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def _jpeg_width(data: bytes):
    """Pixel width from a JPEG header (SOF marker) without decoding the image; None if not found."""
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker in (0xFF, 0x01) or 0xD0 <= marker <= 0xD7:      # padding / markers without a length
            i += 1 if marker == 0xFF else 2
            continue
        seg = int.from_bytes(data[i + 2:i + 4], "big")
        if marker in _SOF:
            return int.from_bytes(data[i + 7:i + 9], "big")
        i += 2 + seg
    return None


def make_thumbnail(root, rel, width) -> bytes:
    """Small JPEG of a capture, made in memory for each request - nothing is written to disk.

    The JPEG is decoded at 1/2, 1/4 or 1/8 size when that is still at least `width` wide (libjpeg scales while
    decoding), which makes a thumbnail of a full-HD frame several times cheaper than a full decode + resize."""
    src = safe_capture_path(root, rel)
    if src.suffix.lower() != ".jpg" or not src.is_file():
        raise FileNotFoundError(rel)
    width = max(32, min(1280, int(width)))
    data = src.read_bytes()
    full_w = _jpeg_width(data)
    flag = cv2.IMREAD_COLOR
    if full_w:
        flag = next((f for k, f in _REDUCED if full_w // k >= width), cv2.IMREAD_COLOR)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), flag)
    if img is None:
        raise FileNotFoundError(rel)
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise OSError(f"could not encode a thumbnail of {rel}")
    return buf.tobytes()
