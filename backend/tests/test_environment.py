"""Jetson: TensorRT, CUDA torch and the GStreamer OpenCV come from JetPack's SYSTEM packages. A venv created without
--system-site-packages (or pip-installed torch/opencv) cannot see them, and models then 'do not load'."""
import app as appmod
import env_check
import model_diag
import settings


def facts(**over):
    base = {"machine": "aarch64", "in_venv": True, "system_site_packages": True,
            "torch": {"version": "2.11.0", "cuda": True, "gpu": "Orin"},
            "tensorrt": {"version": "10.3.0"},
            "opencv": {"version": "4.8.0", "gstreamer": True, "ffmpeg": True},
            "ultralytics": "8.4.30", "lap": True}
    base.update(over)
    return base


def test_a_correct_jetson_environment_has_no_warnings():
    assert env_check.build_warnings(facts()) == []


def test_venv_without_system_site_packages_on_a_jetson_is_called_out():
    w = " ".join(env_check.build_warnings(facts(system_site_packages=False,
                                                torch={"version": "2.14.1+cu130", "cuda": False, "gpu": None},
                                                tensorrt={"error": "No module named 'tensorrt'"})))
    assert "--system-site-packages" in w
    assert "cannot use the gpu" in w.lower() and "tensorrt" in w.lower()


def test_torch_without_cuda_explains_engines_cannot_run():
    w = " ".join(env_check.build_warnings(facts(torch={"version": "2.8.0", "cuda": False, "gpu": None})))
    assert "cpu" in w.lower() and ".engine" in w


def test_missing_torch_is_reported():
    assert any("not installed" in x.lower() for x in env_check.build_warnings(facts(torch={"error": "No module named 'torch'"})))


def test_pip_opencv_hiding_the_gstreamer_build_is_reported():
    w = " ".join(env_check.build_warnings(facts(opencv={"version": "4.11.0", "gstreamer": False, "ffmpeg": True})))
    assert "pip uninstall opencv-python" in w


def test_opencv_without_ffmpeg_is_reported_because_rtsp_needs_it():
    w = " ".join(env_check.build_warnings(facts(opencv={"version": "4.8.0", "gstreamer": True, "ffmpeg": False})))
    assert "ffmpeg" in w.lower() and "rtsp" in w.lower()


def test_missing_lap_is_reported_because_the_tracker_needs_it():
    assert any("lapx" in x for x in env_check.build_warnings(facts(lap=False)))


def test_a_healthy_x86_workstation_without_tensorrt_has_no_warnings():
    f = facts(machine="AMD64", tensorrt={"error": "No module named 'tensorrt'"},
              opencv={"version": "4.11.0", "gstreamer": False, "ffmpeg": True}, system_site_packages=False)
    assert env_check.build_warnings(f) == []


def test_collect_reports_the_real_environment():
    f = env_check.collect()
    for key in ("python", "machine", "in_venv", "system_site_packages", "torch", "tensorrt", "opencv",
                "ultralytics", "lap", "warnings"):
        assert key in f
    assert isinstance(f["warnings"], list)


def test_missing_tensorrt_error_points_at_the_venv_cause():
    msg = model_diag.explain_model_error(ModuleNotFoundError("No module named 'tensorrt'"))
    assert "--system-site-packages" in msg and "JetPack" in msg


def test_environment_endpoint(tmp_path):
    cfg = tmp_path / "config.ini"
    settings.save(cfg, {"MODELS_DIR": str(tmp_path), "OUTPUT_DIR": "c"})
    c = appmod.create_app(cfg, tmp_path / "d", environment_probe=lambda: {"python": "py", "warnings": ["w1"]}).test_client()
    assert c.get("/api/environment").get_json() == {"python": "py", "warnings": ["w1"]}
