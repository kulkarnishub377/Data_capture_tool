"""config.ini loading/saving with typed values and defaults."""
import configparser
from pathlib import Path

# key -> (section, default)
DEFAULTS = {
    "MODELS_DIR": ("Paths", "../models"),
    "OUTPUT_DIR": ("Paths", "./captures"),
    "MODEL": ("Model", "yolov8s.pt"),
    "DEVICE": ("Model", "auto"),
    "HALF": ("Model", True),
    "CONF_THRESH": ("Model", 0.25),
    "IMG_SIZE": ("Model", 640),
    "CLASSES": ("Model", []),
    "SAVE_MODE": ("Capture", "frame"),
    "SAVE_METADATA": ("Capture", False),
    "WHEN": ("Capture", "first"),
    "MIN_BOX": ("Capture", 40),
    "MIN_ROI_OVERLAP": ("Capture", 0.5),
    "JPEG_QUALITY": ("Capture", 95),
    "PAD_NORMAL": ("Capture", 0.15),
    "PAD_BIKE": ("Capture", 0.40),
    "DEDUP_SECONDS": ("Capture", 30),
}
CHOICES = {"SAVE_MODE": {"frame", "crop", "both"}, "WHEN": {"first", "best"}}


def _validate(key, value):
    if key in CHOICES and value not in CHOICES[key]:
        raise ValueError(f"{key} must be one of {sorted(CHOICES[key])}, got {value!r}")


def _to_text(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def _parse(key, raw):
    default = DEFAULTS[key][1]
    raw = raw.split(";")[0].strip() if isinstance(raw, str) else raw
    if isinstance(default, bool):
        return str(raw).lower() in ("1", "true", "yes", "on")
    if isinstance(default, list):
        return [c.strip() for c in str(raw).split(",") if c.strip()]
    if isinstance(default, int):
        return int(float(raw))
    if isinstance(default, float):
        return float(raw)
    return str(raw)


def load(path) -> dict:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    path = Path(path)
    if path.exists():
        cp.read(path, encoding="utf-8")
    out = {}
    for key, (section, default) in DEFAULTS.items():
        if cp.has_option(section, key):
            try:
                out[key] = _parse(key, cp.get(section, key))
                _validate(key, out[key])
                continue
            except ValueError:
                pass
        out[key] = list(default) if isinstance(default, list) else default
    return out


def save(path, updates: dict) -> None:
    """Persist known keys only; other existing keys are preserved."""
    for key, value in updates.items():
        if key in DEFAULTS:
            _validate(key, value)
    path = Path(path)
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    if path.exists():
        cp.read(path, encoding="utf-8")
    for key, value in updates.items():
        if key not in DEFAULTS:
            continue
        section = DEFAULTS[key][0]
        if not cp.has_section(section):
            cp.add_section(section)
        cp.set(section, key, _to_text(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        cp.write(f)
    tmp.replace(path)
