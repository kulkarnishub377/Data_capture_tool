"""MODELS_DIR / OUTPUT_DIR: ~ (home), relative-to-config and absolute paths, plus a visible 'folder not found' report."""
import os
from pathlib import Path

import app as appmod
import settings


def _home(monkeypatch, home):
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))      # Windows reads this one


def test_tilde_means_the_home_folder(monkeypatch, tmp_path):
    _home(monkeypatch, tmp_path / "home")
    got = settings.resolve_path(tmp_path / "cfgdir", "~/_VIDES/_backend_service_combined/models")
    assert got == (tmp_path / "home" / "_VIDES" / "_backend_service_combined" / "models").resolve()


def test_relative_path_is_relative_to_the_config_folder(tmp_path):
    assert settings.resolve_path(tmp_path / "proj", "../models") == (tmp_path / "models").resolve()


def test_absolute_path_is_kept(tmp_path):
    assert settings.resolve_path(tmp_path / "proj", str(tmp_path / "abs")) == (tmp_path / "abs").resolve()


def test_environment_variables_are_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDES_ROOT", str(tmp_path / "v"))
    var = "%VIDES_ROOT%" if os.name == "nt" else "$VIDES_ROOT"
    assert settings.resolve_path(tmp_path, var + "/models") == (tmp_path / "v" / "models").resolve()


def _app(tmp_path, models_dir):
    cfg = tmp_path / "config.ini"
    settings.save(cfg, {"MODELS_DIR": models_dir, "OUTPUT_DIR": "captures"})
    return appmod.create_app(cfg, tmp_path / "data").test_client()


def test_models_are_listed_from_a_home_relative_folder(monkeypatch, tmp_path):
    home = tmp_path / "home"
    models = home / "_VIDES" / "_backend_service_combined" / "models"
    models.mkdir(parents=True)
    (models / "bestv10_s.pt").write_bytes(b"x")
    _home(monkeypatch, home)
    c = _app(tmp_path / "project", "~/_VIDES/_backend_service_combined/models")
    assert c.get("/api/models").get_json() == ["bestv10_s.pt"]
    assert c.put("/api/settings", json={"MODEL": "bestv10_s.pt"}).status_code == 200


def test_paths_endpoint_reports_the_resolved_folder_and_whether_it_exists(monkeypatch, tmp_path):
    _home(monkeypatch, tmp_path / "home")
    c = _app(tmp_path / "project", "~/_VIDES/_backend_service_combined/models")
    r = c.get("/api/paths").get_json()
    expected = (tmp_path / "home" / "_VIDES" / "_backend_service_combined" / "models").resolve()
    assert Path(r["models_dir"]) == expected
    assert r["models_dir_exists"] is False and r["models_count"] == 0
    expected.mkdir(parents=True)
    (expected / "a.pt").write_bytes(b"x")
    r = c.get("/api/paths").get_json()
    assert r["models_dir_exists"] is True and r["models_count"] == 1
    assert Path(r["output_dir"]).name == "captures"
