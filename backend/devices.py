"""Compute-device detection, parsing and per-camera assignment."""


_cache = {}


def detect() -> list:
    """Detected CUDA GPUs as [{id, name, memory_mb}]; empty without CUDA. Cached (importing torch is slow)."""
    if "gpus" not in _cache:
        _cache["gpus"] = _probe()
    return list(_cache["gpus"])


def _probe() -> list:
    try:
        import torch
        import gpu_lock
        with gpu_lock.guard("cuda"):
            if not torch.cuda.is_available():
                return []
            out = []
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                out.append({"id": f"cuda:{i}", "name": p.name, "memory_mb": p.total_memory // (1024 * 1024)})
            return out
    except Exception:
        return []


def resolve(spec: str, available: list):
    """Return (device pool, warnings) for a DEVICE setting."""
    spec = (spec or "auto").strip().lower()
    if spec == "auto":
        return (list(available) or ["cpu"]), []
    if spec == "cpu":
        return ["cpu"], []
    pool, warns = [], []
    for d in [s.strip() for s in spec.split(",") if s.strip()]:
        d = f"cuda:{d}" if d.isdigit() else d
        if d in available:
            pool.append(d)
        else:
            warns.append(f"Requested device {d} is not available")
    if not pool:
        pool = list(available) or ["cpu"]
        warns.append(f"Falling back to {', '.join(pool)}")
    return pool, warns


def assign(camera_ids: list, pool: list, overrides: dict) -> dict:
    """Round-robin cameras over the pool; a valid per-camera override wins."""
    out, i = {}, 0
    for cid in camera_ids:
        ov = overrides.get(cid)
        if ov and ov in pool:
            out[cid] = ov
        else:
            out[cid] = pool[i % len(pool)]
            i += 1
    return out
