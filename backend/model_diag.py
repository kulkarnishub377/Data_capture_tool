"""Turn model-loading failures into messages a person can act on (the service runs on a headless Jetson)."""
import re
from pathlib import Path

MODEL_EXTS = (".pt", ".engine", ".onnx")        # PyTorch, TensorRT (fast on Jetson), ONNX


class ModelFileError(Exception):
    """The file itself is wrong (missing, empty, Git LFS pointer...). The message is already user-ready."""


def model_file_problem(path):
    """Why this file cannot possibly be a model, or None if it looks plausible."""
    p = Path(path)
    if not p.is_file():
        return f"Model file not found: {p}"
    size = p.stat().st_size
    if size == 0:
        return f"Model file is empty (0 bytes): {p}. Copy it again."
    if size < 4096:
        if p.read_bytes()[:64].startswith(b"version https://git-lfs"):
            return (f"{p.name} is a Git LFS pointer, not the model. The real file was not downloaded: run "
                    "'git lfs pull' or copy the actual model file to the device.")
        return f"{p.name} is only {size} bytes - too small to be a model (truncated copy?). Copy it again."
    return None


def explain_model_error(exc) -> str:
    """The error plus a plain-language hint for the usual causes."""
    if isinstance(exc, ModelFileError):
        return str(exc)
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    hints = []
    if "invalid load key" in low:
        hints.append("The file is not a PyTorch model - often a Git LFS pointer or a corrupted copy. Copy the real file again.")
    if "pytorchstreamreader" in low or "central directory" in low:
        hints.append("The file looks corrupt or truncated. Copy it again and compare the size.")
    if "weights only load failed" in low or "unsupported global" in low:
        hints.append("PyTorch refused to unpickle the model. Update ultralytics: pip install -U ultralytics")
    if "can't get attribute" in low or ("ultralytics" in low and "has no attribute" in low):
        hints.append("The model was trained with a newer Ultralytics than the one installed here. "
                     "Update it: pip install -U ultralytics")
    m = re.search(r"no module named '([\w.]+)'", low)
    if m and m.group(1).split(".")[0] != "tensorrt":
        name = m.group(1).split(".")[0]
        hints.append(f"Install the missing package in the Python environment that runs the service (activate its venv): "
                     f"pip install {name}. On a Jetson install torch/torchvision from NVIDIA's JetPack wheels.")
    if "no module named 'tensorrt'" in low:
        hints.append("TensorRT is part of JetPack (a system package). This Python cannot see it - create the virtualenv with "
                     "python3 -m venv --system-site-packages venv and do not pip install tensorrt/torch/opencv-python.")
    elif "libnvinfer" in low or "tensorrt" in low:
        hints.append("TensorRT problem: a .engine file only works with the TensorRT version and GPU it was built on - "
                     "export it again on this device, or use the .pt file.")
    if "stream is capturing" in low or "streamcapture" in low:
        hints.append("Two threads used the GPU at the same time (a CUDA graph was being captured while another camera ran "
                     "inference). The service serialises GPU access, so restart it; if it keeps happening, report it.")
    elif "cuda" in low or "cudnn" in low:
        hints.append("CUDA problem: the installed PyTorch cannot use this GPU (JetPack/PyTorch mismatch) or the GPU is busy. "
                     "Set DEVICE = cpu in config.ini to confirm the model itself is fine, then fix the PyTorch install.")
    return text + (" - " + " ".join(hints) if hints else "")
