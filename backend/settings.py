"""config.ini loading/saving with typed values and defaults."""
import configparser
import re
from pathlib import Path

# key -> (section, default)
DEFAULTS = {
    "DEBUG": ("App", True),
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
    "WHEN": ("Capture", "best"),
    "MIN_BOX": ("Capture", 40),
    "MIN_ROI_OVERLAP": ("Capture", 0.5),
    "JPEG_QUALITY": ("Capture", 95),
    "PAD_NORMAL": ("Capture", 0.15),
    "PAD_BIKE": ("Capture", 0.40),
    "DEDUP_SECONDS": ("Capture", 2),
    "MAX_HOLD_SECONDS": ("Capture", 10),
}
CHOICES = {"SAVE_MODE": {"frame", "crop", "both"}, "WHEN": {"first", "best"}}
RANGES = {  # key -> (min, max), inclusive
    "CONF_THRESH": (0.01, 1.0), "MIN_ROI_OVERLAP": (0.0, 1.0), "MIN_BOX": (1, 5000),
    "DEDUP_SECONDS": (0, 120), "MAX_HOLD_SECONDS": (1, 600), "JPEG_QUALITY": (1, 100), "IMG_SIZE": (160, 2048),
    "PAD_NORMAL": (0.0, 2.0), "PAD_BIKE": (0.0, 2.0),
}


def _validate(key, value):
    if key in CHOICES and value not in CHOICES[key]:
        raise ValueError(f"{key} must be one of {sorted(CHOICES[key])}, got {value!r}")
    if key in RANGES:
        lo, hi = RANGES[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not lo <= value <= hi:
            raise ValueError(f"{key} must be a number between {lo} and {hi}, got {value!r}")


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


def _set_line(lines: list, section: str, key: str, text: str) -> list:
    """Replace `key = ...` inside [section] (keeping comments), or insert/append it."""
    header = f"[{section.lower()}]"
    sec_start = next((i for i, ln in enumerate(lines) if ln.strip().lower() == header), None)
    new = f"{key} = {text}".rstrip() if text else f"{key} ="
    if sec_start is None:
        return lines + ([""] if lines and lines[-1].strip() else []) + [f"[{section}]", new]
    end = len(lines)
    for i in range(sec_start + 1, len(lines)):
        if lines[i].strip().startswith("["):
            end = i
            break
    pat = re.compile(rf"^\s*{re.escape(key)}\s*[=:]", re.IGNORECASE)
    for i in range(sec_start + 1, end):
        if pat.match(lines[i]):
            lines[i] = new
            return lines
    last = end - 1
    while last > sec_start and not lines[last].strip():
        last -= 1
    lines.insert(last + 1, new)
    return lines


def save(path, updates: dict) -> None:
    """Persist known keys only. Edits lines in place so comments/other keys survive."""
    for key, value in updates.items():
        if key in DEFAULTS:
            _validate(key, value)
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for key, value in updates.items():
        if key in DEFAULTS:
            lines = _set_line(lines, DEFAULTS[key][0], key, _to_text(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)
