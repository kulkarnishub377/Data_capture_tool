"""RTSP decoding: GStreamer (Jetson hardware decoder) when OpenCV has it, FFmpeg as the fallback."""
import cv2
import pytest

import camera_io
import env_check
import settings

URL = "rtsp://admin:Arya@123@1.2.3.4:554/Streaming/Channels/101"


# ------------------------------------------------------------------ the URL must survive GStreamer's parser
def test_special_characters_in_the_password_are_percent_encoded():
    assert camera_io.sanitize_rtsp_url(URL) == "rtsp://admin:Arya%40123@1.2.3.4:554/Streaming/Channels/101"


def test_already_encoded_credentials_are_not_double_encoded():
    u = "rtsp://admin:Arya%40123@1.2.3.4/s"
    assert camera_io.sanitize_rtsp_url(u) == u


def test_urls_without_credentials_are_unchanged():
    assert camera_io.sanitize_rtsp_url("rtsp://1.2.3.4:554/s") == "rtsp://1.2.3.4:554/s"
    assert camera_io.sanitize_rtsp_url("rtsp://admin@1.2.3.4/s") == "rtsp://admin@1.2.3.4/s"


# ------------------------------------------------------------------ the pipeline
def test_pipeline_uses_the_hardware_decoder_like_the_working_backend():
    p = camera_io.build_gst_pipeline(URL)
    assert 'rtspsrc location="rtsp://admin:Arya%40123@1.2.3.4:554/Streaming/Channels/101"' in p
    assert "protocols=tcp" in p and "latency=50" in p and "drop-on-latency=true" in p
    assert "rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=true" in p
    assert "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR" in p
    assert p.endswith("appsink drop=true sync=false max-buffers=1")
    assert "Arya@123" not in p


def test_pipeline_options_codec_latency_protocol_decoder():
    p = camera_io.build_gst_pipeline(URL, protocols="udp", latency_ms=200, decoder="nvv4l2decoder", codec="h265")
    assert "protocols=udp" in p and "latency=200" in p and "rtph265depay ! h265parse" in p
    auto = camera_io.build_gst_pipeline(URL, codec="auto")
    assert "decodebin" in auto and "rtph264depay" not in auto


def test_pipeline_cannot_be_hijacked_through_the_url_or_decoder():
    with pytest.raises(ValueError):
        camera_io.build_gst_pipeline('rtsp://h/s" ! filesrc location=/etc/passwd ! fakesink "')
    with pytest.raises(ValueError):
        camera_io.build_gst_pipeline(URL, decoder="nvv4l2decoder ! fakesink")


# ------------------------------------------------------------------ which backend is tried, in which order
class FakeCap:
    def __init__(self, opened=True):
        self.opened = opened

    def isOpened(self):
        return self.opened

    def release(self):
        self.opened = False

    def set(self, *a):
        return True


class Factory:
    """Records every cv2.VideoCapture(...) call; `results` maps backend flag -> opened?"""
    def __init__(self, **results):
        self.calls, self.results = [], results

    def __call__(self, source, flag=None, *params):
        self.calls.append((source, flag))
        return FakeCap(self.results.get(flag, True))


GST, FFM = cv2.CAP_GSTREAMER, cv2.CAP_FFMPEG


def test_auto_uses_gstreamer_when_available_and_stops_there_if_it_opens():
    f = Factory()
    r = camera_io.open_stream(URL, opts={"mode": "auto"}, factory=f, gst_ok=True)
    assert r.backend == "gstreamer" and r.cap.isOpened()
    assert [c[1] for c in f.calls] == [GST] and "nvv4l2decoder" in f.calls[0][0]


def test_auto_falls_back_to_ffmpeg_when_the_pipeline_does_not_open():
    f = Factory(**{str(GST): False})
    f.results = {GST: False, FFM: True}
    r = camera_io.open_stream(URL, opts={"mode": "auto"}, factory=f, gst_ok=True)
    assert r.backend == "ffmpeg" and [c[1] for c in f.calls] == [GST, FFM]
    assert f.calls[1][0] == URL                       # FFmpeg tolerates the raw URL


def test_auto_without_gstreamer_support_goes_straight_to_ffmpeg():
    f = Factory()
    r = camera_io.open_stream(URL, opts={"mode": "auto"}, factory=f, gst_ok=False)
    assert r.backend == "ffmpeg" and [c[1] for c in f.calls] == [FFM]


def test_ffmpeg_mode_never_builds_a_pipeline():
    f = Factory()
    camera_io.open_stream(URL, opts={"mode": "ffmpeg"}, factory=f, gst_ok=True)
    assert [c[1] for c in f.calls] == [FFM]


def test_forced_gstreamer_does_not_fall_back_and_explains_when_unsupported():
    f = Factory()
    r = camera_io.open_stream(URL, opts={"mode": "gstreamer"}, factory=f, gst_ok=False)
    assert r.cap is None and f.calls == [] and "gstreamer" in r.note.lower()
    f2 = Factory()
    f2.results = {GST: False}
    r2 = camera_io.open_stream(URL, opts={"mode": "gstreamer"}, factory=f2, gst_ok=True)
    assert r2.cap is None and [c[1] for c in f2.calls] == [GST]


def test_files_and_non_rtsp_sources_never_use_gstreamer():
    f = Factory()
    r = camera_io.open_stream("video.mp4", is_file=True, opts={"mode": "auto"}, factory=f, gst_ok=True)
    assert r.backend == "file" and [c[1] for c in f.calls] == [FFM]
    f2 = Factory()
    camera_io.open_stream("http://h/stream.mjpg", opts={"mode": "auto"}, factory=f2, gst_ok=True)
    assert [c[1] for c in f2.calls] == [FFM]


def test_gstreamer_support_is_read_from_the_opencv_build_info():
    assert camera_io.gstreamer_supported("  GStreamer:                   YES (1.20.3)\n") is True
    assert camera_io.gstreamer_supported("  GStreamer:                   NO\n") is False
    assert camera_io.gstreamer_supported("nothing") is False


# ------------------------------------------------------------------ settings
def test_stream_settings_defaults_and_validation(tmp_path):
    cfg = settings.load(tmp_path / "none.ini")
    assert (cfg["STREAM_BACKEND"], cfg["GST_LATENCY_MS"], cfg["GST_CODEC"], cfg["GST_PROTOCOLS"], cfg["GST_DECODER"]) == \
        ("auto", 50, "h264", "tcp", "nvv4l2decoder")
    for bad in ({"STREAM_BACKEND": "vlc"}, {"GST_CODEC": "av1"}, {"GST_PROTOCOLS": "http"},
                {"GST_LATENCY_MS": -1}, {"GST_DECODER": "nvv4l2decoder ! fakesink"}):
        with pytest.raises(ValueError):
            settings.save(tmp_path / "c.ini", bad)
    settings.save(tmp_path / "c.ini", {"STREAM_BACKEND": "gstreamer", "GST_LATENCY_MS": 200, "GST_CODEC": "h265"})
    assert settings.load(tmp_path / "c.ini")["STREAM_BACKEND"] == "gstreamer"


# ------------------------------------------------------------------ the environment report explains the pip-OpenCV cause
def _facts(**over):
    f = {"machine": "aarch64", "in_venv": True, "system_site_packages": True,
         "torch": {"version": "2.11.0", "cuda": True, "gpu": "Orin"}, "tensorrt": {"version": "10.3.0"},
         "opencv": {"version": "4.11.0", "gstreamer": False, "ffmpeg": True}, "ultralytics": "8.4.174", "lap": True,
         "nvdec": True}
    f.update(over)
    return f


def test_pip_opencv_warning_explains_the_cpu_fallback_and_the_fix():
    w = " ".join(env_check.build_warnings(_facts()))
    assert "pip uninstall opencv-python" in w and "FFmpeg" in w and "CPU" in w


def test_hardware_decoder_plugin_missing_is_reported_on_a_jetson():
    f = _facts(opencv={"version": "4.8.0", "gstreamer": True, "ffmpeg": True}, nvdec=False)
    assert any("nvv4l2decoder" in x for x in env_check.build_warnings(f))
    f["nvdec"] = True
    assert env_check.build_warnings(f) == []
