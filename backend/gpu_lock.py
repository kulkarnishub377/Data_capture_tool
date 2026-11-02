"""One process-wide lock for everything that touches the GPU.

Ultralytics 8.4 loads a TensorRT engine with a CUDA-graph warm-up (torch.cuda.graph). While one thread is capturing, any
other thread's CUDA call fails with 'operation not permitted when stream is capturing'. With several cameras (one worker
thread each) that happens as soon as two start together. Loading a model and running inference therefore take this lock
(the working VIDES backend does the same with its gpu_lock). CPU work is never serialised."""
import contextlib
import threading

_LOCK = threading.RLock()


def guard(device):
    """Context manager: the GPU lock for CUDA devices, a no-op for the CPU."""
    return _LOCK if str(device).lower().startswith("cuda") else contextlib.nullcontext()
