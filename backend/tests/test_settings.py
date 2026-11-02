import pytest
import settings


def test_load_fills_defaults(tmp_path):
    cfg = settings.load(tmp_path / "missing.ini")
    assert cfg["SAVE_MODE"] == "frame"
    assert cfg["SAVE_METADATA"] is False
    assert cfg["CONF_THRESH"] == 0.25
    assert cfg["WHEN"] == "first"
    assert cfg["DEVICE"] == "auto"
    assert cfg["CLASSES"] == []


def test_save_round_trip(tmp_path):
    p = tmp_path / "config.ini"
    settings.save(p, {"MODEL": "yolov8s.pt", "SAVE_METADATA": True, "CONF_THRESH": 0.4})
    cfg = settings.load(p)
    assert cfg["MODEL"] == "yolov8s.pt"
    assert cfg["SAVE_METADATA"] is True
    assert cfg["CONF_THRESH"] == 0.4


def test_classes_parsed_to_list(tmp_path):
    p = tmp_path / "config.ini"
    settings.save(p, {"CLASSES": ["car", "bus"]})
    assert settings.load(p)["CLASSES"] == ["car", "bus"]


def test_invalid_save_mode_rejected(tmp_path):
    with pytest.raises(ValueError):
        settings.save(tmp_path / "c.ini", {"SAVE_MODE": "nope"})


def test_unknown_keys_ignored(tmp_path):
    p = tmp_path / "c.ini"
    settings.save(p, {"EVIL": "x", "MODEL": "a.pt"})
    assert "EVIL" not in settings.load(p)
