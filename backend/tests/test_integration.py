"""Real model + real video through the real Flask app and CameraWorker (slow)."""
import time
from pathlib import Path

import cv2
import pytest

import app as appmod
import dedup
import roi
import settings

REPO = Path(__file__).resolve().parents[3]
VIDEO = REPO / "vedio" / "2_GetVIDESVideo.mp4"
MODELS = REPO / "models"

pytestmark = pytest.mark.skipif(not VIDEO.exists(), reason="needs the repo's sample video")

POLY = [[60, 120], [650, 120], [650, 540], [60, 540]]


@pytest.mark.parametrize("MODEL", ["bestv10_s.pt", "yolov8s.pt"])  # custom 7-class and stock COCO
def test_real_model_on_real_video_one_image_per_vehicle_inside_roi(tmp_path, MODEL):
    if not (MODELS / MODEL).exists():
        pytest.skip(f"{MODEL} not available")
    cfg = tmp_path / "config.ini"
    settings.save(cfg, {"MODELS_DIR": str(MODELS), "OUTPUT_DIR": str(tmp_path / "out"), "MODEL": MODEL,
                        "DEVICE": "cpu", "SAVE_MODE": "both", "SAVE_METADATA": True, "CONF_THRESH": 0.3})
    c = appmod.create_app(cfg, tmp_path / "data").test_client()
    cam = c.post("/api/cameras", json={"name": "Vid", "url": str(VIDEO)}).get_json()
    assert c.put(f"/api/cameras/{cam['id']}/roi", json={"points": POLY, "frame_size": [704, 576]}).status_code == 200
    r = c.post(f"/api/capture/{cam['id']}/start")
    assert r.status_code == 200, r.get_json()
    t0 = time.time()
    while time.time() - t0 < 240:
        st = c.get("/api/capture/status").get_json()[cam["id"]]
        if st["state"] in ("stopped", "error"):
            break
        time.sleep(0.5)
    assert st["state"] == "stopped", st

    frames = sorted((tmp_path / "out").rglob("frame/*.jpg"))
    crops = sorted((tmp_path / "out").rglob("crop/*.jpg"))
    assert len(frames) >= 1 and len(frames) == len(crops) == st["saved"]
    assert len(list((tmp_path / "out").rglob("*.json"))) == len(frames) + len(crops)
    # every saved vehicle's centre lies inside the ROI
    import json
    for j in (tmp_path / "out").rglob("frame/*.json"):
        x1, y1, x2, y2 = json.loads(j.read_text())["bbox"]
        assert roi.point_in_polygon(((x1 + x2) / 2, (y1 + y2) / 2), POLY)
    # no near-duplicate crops saved within the dedup window
    hashes = [dedup.phash(cv2.imread(str(p))) for p in crops]
    for i in range(len(hashes)):
        for k in range(i + 1, len(hashes)):
            assert bin(hashes[i] ^ hashes[k]).count("1") > 6, (crops[i].name, crops[k].name)
