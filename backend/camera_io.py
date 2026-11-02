"""Talking to cameras safely: reachability check, hard timeouts, capture opening."""
import os
import socket
import threading
from urllib.parse import urlsplit

import cv2

OPEN_TIMEOUT_MS = 5000
READ_TIMEOUT_MS = 5000
GRAB_TIMEOUT_S = 10.0


class CameraError(Exception):
    """A camera problem with a message that is safe and useful to show the user."""


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


def open_capture(source, is_file=False):
    """cv2.VideoCapture with short open/read timeouts for streams; None if it cannot be opened."""
    if is_file:
        cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
    else:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|analyzeduration;3000000|probesize;3000000")
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS,
                  cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS]
        try:
            cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG, params)
        except (TypeError, cv2.error):
            cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def grab_frame(url, timeout=GRAB_TIMEOUT_S):
    """One frame from a stream/file. Raises CameraError with the reason on failure."""
    ep = url_endpoint(url)
    if ep:
        problem = check_port(ep[0], ep[1])
        if problem:
            raise CameraError(problem)

    def one():
        cap = open_capture(url, is_file=not ep)
        if cap is None:
            return None
        try:
            ok, frame = cap.read()
            return frame if ok else None
        finally:
            cap.release()

    frame = run_with_timeout(one, timeout)
    if frame is None:
        raise CameraError(f"The camera did not respond with a picture within {int(timeout)} s "
                          "(wrong path/credentials, or it is at its connection limit)")
    return frame
