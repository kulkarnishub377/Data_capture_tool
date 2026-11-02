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
2. **ROI** - pick the camera, *Load snapshot*, click to place points (drag to adjust). Use *+ New ROI* to draw
   more than one area on the same camera (a vehicle is captured inside ANY of them), then *Save ROIs*.
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
| `DEDUP_SECONDS` | how long a saved vehicle is remembered after it was last seen (default 2) |
| `DEBUG` | `true` = Flask debug mode + DEBUG logging (default) |

## No duplicates
1. A tracked vehicle is saved once (track lock).
2. A new track overlapping a saved vehicle that is still in view (IoU > 0.5, seen within `DEDUP_SECONDS`) is rejected - covers tracker ID switches.
3. A perceptual-hash check on the vehicle crop rejects near-identical images in that same short window.

## Tests
```
python -m pytest            # unit + integration (integration uses ../vedio and ../models)
```
