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


def test_debug_on_by_default(tmp_path):
    assert settings.load(tmp_path / "none.ini")["DEBUG"] is True


def test_save_preserves_comments_and_unrelated_lines(tmp_path):
    p = tmp_path / "config.ini"
    p.write_text("[Model]\n# my note about the model\nMODEL = a.pt\nCONF_THRESH = 0.25\n\n[Capture]\n"
                 "# frame|crop|both\nSAVE_MODE = frame\n", encoding="utf-8")
    settings.save(p, {"MODEL": "b.pt", "SAVE_METADATA": True})
    text = p.read_text(encoding="utf-8")
    assert "# my note about the model" in text and "# frame|crop|both" in text
    cfg = settings.load(p)
    assert cfg["MODEL"] == "b.pt" and cfg["SAVE_METADATA"] is True and cfg["SAVE_MODE"] == "frame"


def test_dedup_seconds_default_is_short(tmp_path):
    assert settings.load(tmp_path / "none.ini")["DEDUP_SECONDS"] == 2
