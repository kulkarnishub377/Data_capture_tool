"""What can this Python environment actually do? (torch/CUDA, TensorRT, OpenCV backends, tracker dependency)

On an NVIDIA Jetson, CUDA torch, TensorRT and the GStreamer OpenCV are JetPack SYSTEM packages. A venv created without
--system-site-packages, or one where pip installed its own torch / opencv, cannot see them - and TensorRT .engine models
then fail to load. build_warnings() turns the facts into plain instructions."""
import importlib.util
import platform
import re
import sys
from pathlib import Path


def _system_site_packages() -> bool:
    if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        return True                                   # not a venv
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    try:
        for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("include-system-site-packages"):
                return line.split("=", 1)[1].strip().lower() == "true"
    except OSError:
        pass
    return False


def collect() -> dict:
    f = {"python": sys.executable, "version": platform.python_version(), "machine": platform.machine(),
         "in_venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix), "system_site_packages": _system_site_packages()}
    try:
        import torch
        f["torch"] = {"version": torch.__version__, "cuda": bool(torch.cuda.is_available()),
                      "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except Exception as e:  # noqa: BLE001
        f["torch"] = {"error": str(e)}
    try:
        import tensorrt
        f["tensorrt"] = {"version": tensorrt.__version__}
    except Exception as e:  # noqa: BLE001
        f["tensorrt"] = {"error": str(e)}
    try:
        import cv2
        info = cv2.getBuildInformation()
        f["opencv"] = {"version": cv2.__version__,
                       "gstreamer": bool(re.search(r"GStreamer:\s+YES", info)),
                       "ffmpeg": bool(re.search(r"FFMPEG:\s+YES", info))}
    except Exception as e:  # noqa: BLE001
        f["opencv"] = {"error": str(e)}
    try:
        import ultralytics
        f["ultralytics"] = ultralytics.__version__
    except Exception as e:  # noqa: BLE001
        f["ultralytics"] = None
        f["ultralytics_error"] = str(e)
    f["lap"] = importlib.util.find_spec("lap") is not None
    f["warnings"] = build_warnings(f)
    return f


def build_warnings(f: dict) -> list:
    w = []
    jetson = str(f.get("machine", "")).lower() in ("aarch64", "arm64")
    torch, trt, cv = f.get("torch", {}), f.get("tensorrt", {}), f.get("opencv", {})
    venv_fix = ("Recreate the virtualenv so it can see JetPack's packages: "
                "python3 -m venv --system-site-packages venv   (and do not pip install torch, torchvision, tensorrt or opencv-python).")
    if jetson and f.get("in_venv") and not f.get("system_site_packages"):
        w.append("This virtualenv was created without --system-site-packages, so it cannot see JetPack's CUDA torch, "
                 "TensorRT and GStreamer OpenCV. " + venv_fix)
    if "error" in torch:
        w.append("PyTorch is not installed in this Python environment (" + str(torch["error"]) + ").")
    elif not torch.get("cuda"):
        if jetson:
            w.append("PyTorch cannot use the GPU here (CUDA not available): a pip-installed torch build does not work "
                     "on a Jetson. TensorRT .engine models cannot run and .pt models would run on the CPU. " + venv_fix)
        else:
            w.append("No CUDA GPU is available to PyTorch: models run on the CPU (slow) and TensorRT .engine models cannot run.")
    if jetson and "error" in trt:
        w.append("TensorRT is not visible to this Python (" + str(trt["error"]) + "). It ships with JetPack as a system "
                 "package. " + venv_fix)
    if "error" not in cv:
        if jetson and not cv.get("gstreamer"):
            w.append("This OpenCV was built without GStreamer (probably the pip build hiding JetPack's). "
                     "Run: pip uninstall opencv-python opencv-python-headless opencv-contrib-python")
        if not cv.get("ffmpeg"):
            w.append("This OpenCV has no FFmpeg backend: RTSP streams cannot be opened with it.")
    if f.get("ultralytics") is None:
        w.append("ultralytics is not installed (" + str(f.get("ultralytics_error", "")) + ").")
    if f.get("lap") is False:
        w.append("The tracker needs the 'lap' module: pip install lapx (ultralytics tries to install it at runtime, which fails offline).")
    return w
