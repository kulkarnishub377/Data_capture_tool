"""Talking to cameras safely: reachability check, hard timeouts, and opening a stream.

RTSP is decoded with GStreamer when OpenCV has it (on an NVIDIA Jetson this uses the hardware video decoder, NVDEC,
instead of the CPU) and falls back to FFmpeg when the pipeline does not open. Files and non-RTSP sources always use FFmpeg.
"""
import os
import re
import socket
import threading
from dataclasses import dataclass
from urllib.parse import quote, unquote, urlsplit

import cv2

OPEN_TIMEOUT_MS = 5000
READ_TIMEOUT_MS = 5000
GRAB_TIMEOUT_S = 10.0

DEFAULT_OPTS = {"mode": "auto", "latency_ms": 50, "codec": "h264", "decoder": "nvv4l2decoder", "protocols": "tcp"}
_active_opts = dict(DEFAULT_OPTS)          # used by grab_frame (snapshot / test connection)
_SAFE_TOKEN = re.compile(r"[A-Za-z0-9_\-]+")


class CameraError(Exception):
    """A camera problem with a message that is safe and useful to show the user."""


def configure(**opts):
    """Set the stream options used by grab_frame (the capture workers pass their own)."""
    _active_opts.update({k: v for k, v in opts.items() if k in DEFAULT_OPTS})


def stream_opts(cfg) -> dict:
    """Stream options from a settings dict."""
    return {"mode": cfg["STREAM_BACKEND"], "latency_ms": cfg["GST_LATENCY_MS"], "codec": cfg["GST_CODEC"],
            "decoder": cfg["GST_DECODER"], "protocols": cfg["GST_PROTOCOLS"]}


def run_with_timeout(fn, timeout):
    """Run fn() in a daemon thread; return its result, or None if it did not finish in time."""
    box = {}

    def target():
        try:
            box["v"] = fn()
        except Exception as e:  # noqa: BLE001
            box["e"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive() or "e" in box:
        return None
    return box.get("v")


def url_endpoint(url):
    """(host, port) for a network stream URL, or None for files/devices."""
    try:
        parts = urlsplit(str(url))
        if parts.scheme.lower() not in ("rtsp", "rtsps", "http", "https", "rtmp"):
            return None
        host = parts.hostname
        if not host:
            return None
        default = {"rtsp": 554, "rtsps": 322, "http": 80, "https": 443, "rtmp": 1935}[parts.scheme.lower()]
        return host, parts.port or default
    except ValueError:
        return None


def check_port(host, port, timeout=4.0):
    """None if a TCP connection works; otherwise a human-readable reason."""
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, int(port)))
        return None
    except socket.timeout:
        return f"Cannot reach {host}:{port} (timed out) - the camera is offline, the port is wrong or it is blocked"
    except ConnectionRefusedError:
        return f"Cannot reach {host}:{port} (connection refused) - nothing is listening on that port"
    except OSError as e:
        return f"Cannot reach {host}:{port} ({e.strerror or type(e).__name__})"
    finally:
        s.close()


# ---------------------------------------------------------------------------------------------- GStreamer
def sanitize_rtsp_url(url: str) -> str:
    """Percent-encode the user/password so characters like '@' in a password do not break GStreamer's URL parser."""
    try:
        scheme_end = url.index("://") + 3
        rest = url[scheme_end:]
        last_at = rest.rfind("@")
        if last_at == -1:
            return url
        credentials, host_part = rest[:last_at], rest[last_at + 1:]
        user, pwd = credentials.split(":", 1) if ":" in credentials else (credentials, "")
        user, pwd = quote(unquote(user), safe=""), quote(unquote(pwd), safe="")
        auth = f"{user}:{pwd}@" if pwd else f"{user}@"
        return f"{url[:scheme_end]}{auth}{host_part}"
    except ValueError:
        return url


def build_gst_pipeline(url, protocols="tcp", latency_ms=50, decoder="nvv4l2decoder", codec="h264") -> str:
    """RTSP -> hardware decode -> BGR frames for OpenCV (appsink keeps only the newest frame)."""
    safe_url = sanitize_rtsp_url(str(url))
    if re.search(r'[\s"\\]', safe_url):
        raise ValueError("camera URL contains characters that cannot be used in a GStreamer pipeline")
    if not _SAFE_TOKEN.fullmatch(str(decoder)):
        raise ValueError(f"invalid GStreamer decoder name: {decoder!r}")
    proto = "udp" if str(protocols).lower() == "udp" else "tcp"
    latency = max(0, min(5000, int(latency_ms)))
    src = (f'rtspsrc location="{safe_url}" protocols={proto} latency={latency} drop-on-latency=true '
           f'tcp-timeout={OPEN_TIMEOUT_MS * 1000} timeout={OPEN_TIMEOUT_MS * 1000} ! ')
    perf = " enable-max-performance=true" if decoder == "nvv4l2decoder" else ""
    if codec == "auto":
        mid = "decodebin ! "                                   # picks the right depay/parse/decoder (H.264 or H.265)
    else:
        c = "h265" if codec == "h265" else "h264"
        mid = f"rtp{c}depay ! {c}parse ! {decoder}{perf} ! "
    tail = ("nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true sync=false max-buffers=1")
    return src + mid + tail


_gst_cache = {}


def gstreamer_supported(build_info=None) -> bool:
    """Does this OpenCV have GStreamer? (reads cv2.getBuildInformation(), cached for the real call)"""
    if build_info is not None:
        return bool(re.search(r"GStreamer:\s+YES", build_info))
    if "v" not in _gst_cache:
        _gst_cache["v"] = gstreamer_supported(cv2.getBuildInformation())
    return _gst_cache["v"]


# ---------------------------------------------------------------------------------------------- opening
@dataclass
class OpenResult:
    cap: object          # an opened cv2.VideoCapture, or None
    backend: str         # "gstreamer" | "ffmpeg" | "file" | None
    note: str = ""       # why it failed / what was tried


def _is_rtsp(source) -> bool:
    return str(source).lower().startswith(("rtsp://", "rtsps://"))


def open_stream(source, is_file=False, opts=None, factory=None, gst_ok=None) -> OpenResult:
    """Open `source`: GStreamer for RTSP when allowed and supported, otherwise (or on failure) FFmpeg."""
    o = {**_active_opts, **(opts or {})}
    factory = factory or cv2.VideoCapture
    mode = o["mode"]
    notes = []
    if is_file or not _is_rtsp(source):
        order = ["ffmpeg"]
    else:
        gst = gstreamer_supported() if gst_ok is None else gst_ok
        if mode == "ffmpeg":
            order = ["ffmpeg"]
        elif mode == "gstreamer":
            if not gst:
                return OpenResult(None, None, "GStreamer was requested (STREAM_BACKEND = gstreamer) but this OpenCV was built "
                                              "without GStreamer support - see Settings > Check environment")
            order = ["gstreamer"]
        else:
            order = (["gstreamer"] if gst else []) + ["ffmpeg"]

    for backend in order:
        if backend == "gstreamer":
            try:
                pipeline = build_gst_pipeline(source, o["protocols"], o["latency_ms"], o["decoder"], o["codec"])
            except ValueError as e:
                notes.append(str(e))
                continue
            cap = factory(pipeline, cv2.CAP_GSTREAMER)
            if cap is not None and cap.isOpened():
                return OpenResult(cap, "gstreamer")
            if cap is not None:
                cap.release()
            notes.append("the GStreamer pipeline did not open")
            continue
        if not is_file:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|analyzeduration;3000000|probesize;3000000")
        if is_file or not _is_rtsp(source):
            cap = factory(str(source), cv2.CAP_FFMPEG)
        else:
            params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS, cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS]
            try:
                cap = factory(str(source), cv2.CAP_FFMPEG, params)
            except (TypeError, cv2.error):
                cap = factory(str(source), cv2.CAP_FFMPEG)
        if cap is not None and cap.isOpened():
            return OpenResult(cap, "file" if is_file else "ffmpeg")
        if cap is not None:
            cap.release()
        notes.append("FFmpeg could not open the stream")
    return OpenResult(None, None, "; ".join(notes))


def open_capture(source, is_file=False, opts=None):
    """cv2.VideoCapture for `source`, or None if it cannot be opened."""
    return open_stream(source, is_file, opts).cap


def grab_frame(url, timeout=GRAB_TIMEOUT_S):
    """One frame from a stream/file. Raises CameraError with the reason on failure."""
    ep = url_endpoint(url)
    if ep:
        problem = check_port(ep[0], ep[1])
        if problem:
            raise CameraError(problem)
    why = {}

    def one():
        r = open_stream(url, is_file=not ep)
        why["note"] = r.note
        if r.cap is None:
            return None
        try:
            ok, frame = r.cap.read()
            return frame if ok else None
        finally:
            r.cap.release()

    frame = run_with_timeout(one, timeout)
    if frame is None:
        extra = f" ({why['note']})" if why.get("note") else ""
        raise CameraError(f"The camera did not respond with a picture within {int(timeout)} s{extra} "
                          "(wrong path/credentials, or it is at its connection limit)")
    return frame
