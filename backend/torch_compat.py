"""PyTorch >= 2.6 loads checkpoints with weights_only=True by default, which refuses the pickled model classes inside
a trained Ultralytics .pt and fails with 'Weights only load failed / Unsupported global'. These are our own trusted
model files, so restore the old default - exactly what the annotator tool does. Must run BEFORE ultralytics is imported."""


def apply() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch missing: the model load will report that itself
        return False
    if getattr(torch.load, "_dct_patched", False):
        return True
    original = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    load._dct_patched = True
    torch.load = load
    return True
