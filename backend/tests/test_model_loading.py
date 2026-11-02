"""Models must load on the deployment device (Jetson), and when they do not the reason must be visible."""
import pickle

import pytest

import app as appmod
import model_diag
import settings


# ------------------------------------------------------------------ PyTorch >= 2.6 refuses to unpickle trained checkpoints
class _Trained:                      # stands in for an Ultralytics model object inside a .pt checkpoint
    pass


def test_checkpoints_load_with_the_default_torch_load(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import torch_compat
    monkeypatch.setattr(torch, "load", torch.load)          # undo the patch after the test
    path = tmp_path / "m.pt"
    torch.save({"model": _Trained()}, path)
    torch_compat.apply()
    assert "model" in torch.load(path)                      # default arguments, as ultralytics calls it
    torch_compat.apply()                                    # idempotent
    assert getattr(torch.load, "_dct_patched", False)


# ------------------------------------------------------------------ is the FILE a model at all?
def test_missing_empty_and_git_lfs_pointer_files_are_recognised(tmp_path):
    assert "not found" in model_diag.model_file_problem(tmp_path / "nope.pt").lower()
    empty = tmp_path / "empty.pt"
    empty.write_bytes(b"")
    assert "empty" in model_diag.model_file_problem(empty).lower()
    lfs = tmp_path / "lfs.pt"
    lfs.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 12345678\n")
    assert "git lfs" in model_diag.model_file_problem(lfs).lower()
    tiny = tmp_path / "tiny.pt"
    tiny.write_bytes(b"x" * 100)
    assert "too small" in model_diag.model_file_problem(tiny).lower()
    ok = tmp_path / "ok.pt"
    ok.write_bytes(b"x" * 20000)
    assert model_diag.model_file_problem(ok) is None


# ------------------------------------------------------------------ errors that people can act on
@pytest.mark.parametrize("exc,expect", [
    (pickle.UnpicklingError("invalid load key, 'v'."), "git lfs"),
    (RuntimeError("PytorchStreamReader failed reading zip archive: failed finding central directory"), "corrupt"),
    (AttributeError("Can't get attribute 'C3k2' on <module 'ultralytics.nn.modules.block'>"), "pip install -U ultralytics"),
    (ModuleNotFoundError("No module named 'ultralytics'"), "install"),
    (RuntimeError("CUDA error: no kernel image is available for execution on the device"), "cuda"),
    (ImportError("libnvinfer.so.8: cannot open shared object file"), "tensorrt"),
    (ModuleNotFoundError("No module named 'onnxruntime'"), "onnxruntime"),
])
def test_known_load_errors_get_a_plain_language_hint(exc, expect):
    msg = model_diag.explain_model_error(exc)
    assert expect.lower() in msg.lower() and type(exc).__name__ in msg


def test_unknown_errors_still_show_the_original_message():
    assert "something odd" in model_diag.explain_model_error(RuntimeError("something odd"))


# ------------------------------------------------------------------ the API
def _client(tmp_path, names_loader=None, prober=None):
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    cfg = tmp_path / "config.ini"
    settings.save(cfg, {"MODELS_DIR": str(models), "OUTPUT_DIR": "captures", "MODEL": "a.pt"})
    kw = {}
    if names_loader:
        kw["names_loader"] = names_loader
    if prober:
        kw["model_prober"] = prober
    return appmod.create_app(cfg, tmp_path / "data", **kw).test_client(), models


def test_dropdown_lists_pt_engine_and_onnx_models(tmp_path):
    c, models = _client(tmp_path)
    for n in ("b.engine", "a.pt", "c.onnx", "notes.txt", "d.pt.bak"):
        (models / n).write_bytes(b"x" * 10)
    assert c.get("/api/models").get_json() == ["a.pt", "b.engine", "c.onnx"]


def test_a_git_lfs_pointer_is_reported_with_the_real_reason(tmp_path):
    c, models = _client(tmp_path)
    (models / "a.pt").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 99\n")
    r = c.get("/api/model/classes?model=a.pt")
    assert r.status_code == 400 and "git lfs" in r.get_json()["error"].lower()


def test_a_loader_failure_comes_back_with_a_hint(tmp_path):
    def loader(path):
        raise AttributeError("Can't get attribute 'C3k2' on <module 'ultralytics.nn.modules.block'>")

    c, models = _client(tmp_path, loader)
    (models / "a.pt").write_bytes(b"x" * 20000)
    r = c.get("/api/model/classes?model=a.pt")
    assert r.status_code == 400 and "pip install -U ultralytics" in r.get_json()["error"]


def test_model_test_endpoint_reports_success(tmp_path):
    c, models = _client(tmp_path, prober=lambda path, device, half, imgsz: {0: "car", 1: "bus"})
    (models / "a.pt").write_bytes(b"x" * 20000)
    r = c.post("/api/model/test?model=a.pt").get_json()
    assert r["ok"] is True and r["classes"] == 2 and r["model"] == "a.pt" and r["seconds"] >= 0
    assert r["device"] == "cpu" and "test frame" in r["message"]


def test_model_test_endpoint_reports_failure_without_an_http_error(tmp_path):
    def prober(path, device, half, imgsz):
        raise RuntimeError("CUDA error: no kernel image is available")

    c, models = _client(tmp_path, prober=prober)
    (models / "a.pt").write_bytes(b"x" * 20000)
    res = c.post("/api/model/test?model=a.pt")
    body = res.get_json()
    assert res.status_code == 200 and body["ok"] is False and "cuda" in body["error"].lower()


def test_model_test_for_a_missing_file_says_so(tmp_path):
    c, _ = _client(tmp_path)
    body = c.post("/api/model/test?model=ghost.pt").get_json()
    assert body["ok"] is False and "not found" in body["error"].lower()


def test_start_reports_the_hint_too(tmp_path):
    def loader(path):
        raise ModuleNotFoundError("No module named 'ultralytics'")

    c, models = _client(tmp_path, loader)
    (models / "a.pt").write_bytes(b"x" * 20000)
    cam = c.post("/api/cameras", json={"name": "A", "url": "rtsp://x"}).get_json()
    c.put(f"/api/cameras/{cam['id']}/roi", json={"points": [[0, 0], [9, 0], [9, 9]], "frame_size": [640, 480]})
    r = c.post(f"/api/capture/{cam['id']}/start")
    assert r.status_code == 400 and "install" in r.get_json()["error"].lower()
