"""Ultralytics 8.4 loads a TensorRT engine with a CUDA-graph warm-up. If another camera thread touches the GPU meanwhile:
'CUDA error: operation not permitted when stream is capturing'. All GPU work therefore goes through ONE lock."""
import sys
import threading
import time
import types

import numpy as np
import pytest

import capture_engine as ce
import gpu_lock
import model_diag

POLYS = [[[0, 0], [190, 0], [190, 190], [0, 190]]]
SIZE = (200, 200)


class Probe:
    """Counts how many threads are inside the (fake) GPU at the same moment."""
    def __init__(self):
        self.now = self.peak = 0
        self.lock = threading.Lock()
        self.track_kwargs = []

    def work(self, seconds=0.05):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        time.sleep(seconds)
        with self.lock:
            self.now -= 1


@pytest.fixture
def probe(monkeypatch):
    p = Probe()

    class Result:
        boxes = None

    class FakeYOLO:
        def __init__(self, path, task=None):
            p.work()

        @property
        def names(self):
            p.work()
            return {0: "car"}

        def track(self, crop, **kw):
            p.track_kwargs.append(kw)
            p.work()
            return [Result()]

        def predict(self, *a, **kw):
            p.work()

    mod = types.ModuleType("ultralytics")
    mod.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", mod)
    return p


def _detector(path="m.engine", device="cuda:0", half=True):
    return ce.YoloDetector(path, device, half, 0.3, 640, ["car"], POLYS, SIZE)


def _run_cameras(device, n=3):
    errors = []

    def camera():
        try:
            d = _detector(device=device)
            for _ in range(3):
                d(np.zeros((200, 200, 3), np.uint8))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=camera) for _ in range(n)]
    [t.start() for t in threads]
    [t.join(10) for t in threads]
    assert not errors


def test_gpu_access_from_several_cameras_is_serialised(probe):
    _run_cameras("cuda:0")
    assert probe.peak == 1


def test_cpu_work_is_not_serialised(probe):
    _run_cameras("cpu")
    assert probe.peak >= 2


def test_listing_class_names_and_the_model_test_wait_for_the_gpu(probe):
    for call in (lambda: ce.load_model_names("m.engine"), lambda: ce.probe_model("m.engine", "cuda:0", True, 640)):
        done = threading.Event()
        with gpu_lock.guard("cuda:0"):                      # another camera is using the GPU
            threading.Thread(target=lambda: (call(), done.set()), daemon=True).start()
            assert not done.wait(0.3)                       # must wait, not touch the GPU
        assert done.wait(5)


def test_half_is_only_sent_for_pytorch_models_on_cuda(probe):
    frame = np.zeros((200, 200, 3), np.uint8)
    _detector("m.engine", "cuda:0", True)(frame)            # TensorRT: precision is built into the engine
    _detector("m.pt", "cuda:0", True)(frame)
    _detector("m.pt", "cpu", True)(frame)
    engine, pt_cuda, pt_cpu = probe.track_kwargs
    assert "half" not in engine and pt_cuda["half"] is True and "half" not in pt_cpu


def test_the_guard_for_cpu_never_blocks():
    with gpu_lock.guard("cuda:0"):
        with gpu_lock.guard("cpu"):
            pass


def test_stream_capture_error_is_explained_as_a_concurrency_problem_not_a_broken_install():
    msg = model_diag.explain_model_error(
        RuntimeError("CUDA error: operation not permitted when stream is capturing"))
    assert "same time" in msg.lower() and "device = cpu" not in msg.lower()
