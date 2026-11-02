"""How the server binds (0.0.0.0 for other machines) and the root-relative models folder."""
import app as appmod
import settings


def test_defaults_listen_on_all_interfaces(tmp_path):
    cfg = settings.load(tmp_path / "none.ini")
    assert cfg["HOST"] == "0.0.0.0" and cfg["PORT"] == 5100


def test_network_binding_never_enables_the_interactive_debugger(tmp_path):
    """Flask's debugger console = remote code execution; it must stay off when other machines can reach the server."""
    cfg = settings.load(tmp_path / "none.ini")            # DEBUG on, HOST 0.0.0.0
    o = appmod.server_options(cfg, {})
    assert o["host"] == "0.0.0.0" and o["port"] == 5100
    assert o["debug"] is True and o["use_debugger"] is False and o["use_reloader"] is False


def test_loopback_only_may_use_the_debugger_when_debug_is_on(tmp_path):
    cfg = settings.load(tmp_path / "none.ini")
    cfg["HOST"] = "127.0.0.1"
    assert appmod.server_options(cfg, {})["use_debugger"] is True
    cfg["DEBUG"] = False
    assert appmod.server_options(cfg, {})["use_debugger"] is False


def test_environment_overrides_host_and_port(tmp_path):
    cfg = settings.load(tmp_path / "none.ini")
    o = appmod.server_options(cfg, {"HOST": "127.0.0.1", "PORT": "5999"})
    assert (o["host"], o["port"]) == ("127.0.0.1", 5999)


def test_invalid_port_is_rejected_with_a_clear_message(tmp_path):
    import pytest
    cfg = settings.load(tmp_path / "none.ini")
    with pytest.raises(ValueError, match="PORT"):
        appmod.server_options(cfg, {"PORT": "abc"})
    with pytest.raises(ValueError):
        settings.save(tmp_path / "c.ini", {"PORT": 70000})


def test_lan_addresses_are_ipv4_strings_without_loopback():
    for ip in appmod.lan_addresses():
        assert ip.count(".") == 3 and not ip.startswith("127.")


def test_models_dir_given_from_the_filesystem_root_does_not_crash_when_missing(tmp_path):
    """Deployment: MODELS_DIR = /_VIDES/_backend_service_combined/models (root-relative). Missing -> empty list, no 500."""
    cfg = tmp_path / "config.ini"
    settings.save(cfg, {"MODELS_DIR": "/_VIDES_does_not_exist_here/_backend_service_combined/models",
                        "OUTPUT_DIR": "captures"})
    c = appmod.create_app(cfg, tmp_path / "data").test_client()
    assert c.get("/api/models").get_json() == []
    r = c.put("/api/settings", json={"MODEL": "yolov8s.pt"})
    assert r.status_code == 400 and "not found" in r.get_json()["error"].lower()


def test_root_relative_models_dir_resolves_against_the_drive_of_the_config(tmp_path):
    from pathlib import Path
    base = tmp_path / "config.ini"
    resolved = (base.parent / "/_VIDES/_backend_service_combined/models").resolve()
    assert resolved == (Path(tmp_path.anchor) / "_VIDES" / "_backend_service_combined" / "models").resolve()
