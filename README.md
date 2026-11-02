# Data Capture Tool

Captures **one clean image per vehicle** from live RTSP streams, only while the vehicle is inside a
ROI you draw - to build training data. No database: cameras/ROI live in `data/cameras.json`, images in
`captures/`. Works with custom-trained YOLO models and stock COCO models (e.g. `yolov8s.pt`).

## Run
```
pip install -r requirements.txt
python backend/app.py          # or start.bat  ->  http://127.0.0.1:5100
```

## Folders
| Folder | Contents |
|---|---|
| `backend/` | Flask API, capture engine, dedup, storage, settings, devices, tests |
| `frontend/` | `index.html` + `static/` (css, js) - Live, ROI, Cameras, Gallery |
| `config.ini` | model, device, save options (edit or use the Live tab) |
| `data/` | `cameras.json` (RTSP URLs - git-ignored) |
| `captures/` | `<camera>/<date>/frame|crop/*.jpg` |

## Use
1. **Cameras** - add a name + RTSP URL, press *Test*.
2. **ROI** - pick the camera, *Load snapshot*, click to place points (drag to adjust), *Save ROI*.
3. **Live** - choose model / device / save mode / metadata / classes, press *Start*.
4. **Gallery** - review, filter, enlarge, delete.

## config.ini
| Key | Meaning |
|---|---|
| `MODELS_DIR`, `MODEL` | folder scanned for `*.pt` and the active model |
| `DEVICE` | `auto`, `cpu`, `cuda:0`, or `cuda:0,cuda:1` (cameras spread round-robin over GPUs) |
| `HALF` | FP16 on CUDA |
| `CLASSES` | class names to capture; empty = auto (COCO: car/motorcycle/bus/truck, custom: all) |
| `SAVE_MODE` | `frame` (default), `crop`, or `both` |
| `SAVE_METADATA` | `true` writes a `.json` next to each image; default `false` |
| `WHEN` | `first` (first qualifying frame in ROI) or `best` (best frame before leaving ROI) |
| `DEDUP_SECONDS` | window for the duplicate checks |

## No duplicates
1. A tracked vehicle is saved once (track lock).
2. A new track overlapping a recent save (IoU > 0.5, within `DEDUP_SECONDS`) is rejected - covers tracker ID switches.
3. A perceptual-hash check on the vehicle crop rejects near-identical images in that window.

## Tests
```
python -m pytest            # unit + integration (integration uses ../vedio and ../models)
```
