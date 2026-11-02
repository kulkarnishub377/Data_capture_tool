# Data Capture Tool

Captures **one clean image per vehicle** from live RTSP camera streams, only while the vehicle is inside an area
you draw (an ROI), to build training data for detection models. Everything is local: no database, no cloud.

- **Any YOLO model** - your own trained weights (e.g. 7 vehicle classes) or stock COCO models (`yolov8s.pt`).
- **One image per vehicle, no duplicates** - tracking + duplicate suppression; the next car in the same lane is still captured.
- **Multiple ROIs per camera**, multiple cameras, **multi-GPU** support.
- **Browser UI**: live view with the newest captures beside it, ROI editor, camera manager, gallery, settings.
- **Plain files**: cameras in `data/cameras.json`, images in `captures/`, settings in `config.ini`.

> Using an AI assistant on this repo? Start with [`AGENTS.md`](AGENTS.md) and [`llms.txt`](llms.txt).

---

## Quick start

```bash
cd Data_capture_tool
pip install -r requirements.txt
python backend/app.py            # or double-click start.bat
```

The browser opens at <http://127.0.0.1:5100>. Then:

1. **Cameras** tab - add a name and RTSP URL (`rtsp://user:pass@host:554/stream`), press **Test**.
2. **ROI** tab - choose the camera, **Load snapshot**, click to place polygon points (drag to adjust).
   Use **+ New ROI** to draw more areas on the same camera. **Save ROIs**.
3. **Settings** tab - pick the model, device, classes and thresholds, then **Save & restart**.
4. **Live** tab - press **Start** on a camera (or **Start all**). Captured vehicles appear in **Recent captures** on the right.
   Cameras that cannot start (e.g. no ROI yet) are reported by name and do not stop the others.
5. **Gallery** tab - browse, filter, enlarge and delete everything captured.

**Theme:** dark by default (neutral grey, same palette as the VIDES frontend); the sun/moon button in the header
switches to a light theme and remembers your choice.

Requirements: Python 3.10+ (developed on 3.11/3.13), Windows or Linux. A GPU is optional (CPU works, slower).

---

## The tabs

| Tab | What it does |
|---|---|
| **Live** | Settings summary; a camera toolbar with **Start all / Stop all** and a **Layout** selector (Auto, or 1-4 cameras per row, remembered); one live card per camera (video with ROI and detection boxes, status, counters, **full-screen** button, Start/Stop); and a **Recent captures** panel with thumbnails + class tags. Click a thumbnail to enlarge or delete. |
| **ROI** | Draw one or more polygons per camera on a fresh snapshot. A vehicle is captured when it is inside **any** ROI. |
| **Cameras** | Add / enable / test / delete cameras. Passwords are masked in the UI. |
| **Gallery** | Browse all captures with camera / date / class filters and paging. |
| **Settings** | All model and capture options, grouped in sections. Nothing applies until you press **Save & restart**. |

### Settings (and what Save & restart does)

Edits collect in a draft - rows you changed are highlighted, sections show a dot, the top bar shows
"N unsaved changes". **Save & restart** sends everything in one request; any running camera is restarted
**once** with the new settings. Saved images stay on disk and the saved/skipped counters keep counting.
**Discard** returns every control to the saved values.

| Section | Setting | Meaning |
|---|---|---|
| Model & device | Detection model | Any `*.pt` in `MODELS_DIR`. Switching resets classes to that model's defaults. |
| | Compute device | `auto` (all GPUs, else CPU), `cpu`, `cuda:0`, or a list `cuda:0,cuda:1`. Cameras are spread round-robin over GPUs. |
| Detection | Confidence threshold | Ignore detections below this score (0.01 - 1). |
| | Classes to capture | Searchable picker; "Vehicles" = the model's default set. |
| Capture | Image to save | **Full frame** (default), **Crop**, or **Both**. |
| | Capture moment | **First in ROI** (default) or **Best frame** (largest, most confident frame before the vehicle leaves). |
| | Keep metadata | Off (default) = images only. On = a `.json` next to each image. |
| Advanced | Minimum vehicle size | Shorter side in px (default 40). |
| | Minimum overlap with ROI | 0 - 1 (default 0.5). Stops half-in vehicles being saved. |
| | Duplicate window | Seconds (default 2). See "No duplicates". |
| | JPEG quality | 1 - 100 (default 95). |
| | Detection size | Model input size in px (default 640). |

---

## How capture works

Per camera, three threads run:

1. **Reader** - reads the RTSP stream (TCP) as fast as it arrives and keeps only the **newest frame**, so slow
   inference never makes the picture lag. Reconnects automatically; open/read timeouts are 5 s.
2. **Detect + track** - runs YOLO with ByteTrack on the ROI area (plus a 30% margin, so a vehicle half outside
   the ROI is seen whole). A detection is eligible when: its class is selected, confidence >= threshold, box
   >= min size, its centre is inside an ROI, and >= min overlap of the box is inside the ROIs.
3. **Saver** - writes files from a queue so a slow disk never drops frames.

Each camera loads **its own model instance** (ByteTrack state is per instance) on its assigned device.

### No duplicates

1. **Track lock** - a tracked vehicle is saved at most once.
2. **ID-switch guard** - if the tracker gives the same vehicle a new ID, a new track overlapping a saved vehicle
   that is still in view (IoU > 0.5, seen within the *duplicate window*) is rejected.
3. **Perceptual hash** on the vehicle crop rejects near-identical images inside that same short window.

A saved vehicle is only "remembered" while it keeps being detected plus the duplicate window afterwards, so a
different car arriving later at the same spot is captured normally. File names never collide (`_2`, `_3` suffix).

---

## Files and folders

```
Data_capture_tool/
  backend/            Flask API, capture engine, dedup, storage, settings, devices, tests/
  frontend/           index.html + static/ (css, js, img, vendor/fontawesome)
  config.ini          settings (edited in place by the UI; comments are preserved)
  data/               cameras.json (RTSP URLs - git-ignored), thumbs/ (thumbnail cache)
  captures/           <camera>/<YYYY-MM-DD>/frame|crop/<camera>_<class>_<YYYYmmdd_HHMMSS_mmm>.jpg
  docs/               ARCHITECTURE.md, API.md
  AGENTS.md, llms.txt, llms-full.txt     for AI assistants
```

With **Keep metadata** on, each image gets a same-named `.json`:
`{camera, class, conf, track_id, bbox, timestamp, model, save_mode}`.

## config.ini

Edit the file or use the Settings tab (the UI edits lines in place, so your comments survive).

| Section | Key | Default | Notes |
|---|---|---|---|
| App | `DEBUG` | `true` | Flask debug mode + DEBUG logging (reloader is off). |
| Paths | `MODELS_DIR` | `../models` | Relative to `config.ini`. |
| Paths | `OUTPUT_DIR` | `./captures` | |
| Model | `MODEL` | | file inside `MODELS_DIR` |
| Model | `DEVICE` | `auto` | `auto`, `cpu`, `cuda:0`, `cuda:0,cuda:1` |
| Model | `HALF` | `true` | FP16 on CUDA only |
| Model | `CONF_THRESH` | `0.25` | 0.01 - 1 |
| Model | `IMG_SIZE` | `640` | 160 - 2048 |
| Model | `CLASSES` | empty | class **names**, comma separated; empty = auto |
| Capture | `SAVE_MODE` | `frame` | `frame` / `crop` / `both` |
| Capture | `SAVE_METADATA` | `false` | |
| Capture | `WHEN` | `first` | `first` / `best` |
| Capture | `MIN_BOX` | `40` | 1 - 5000 |
| Capture | `MIN_ROI_OVERLAP` | `0.5` | 0 - 1 |
| Capture | `JPEG_QUALITY` | `95` | 1 - 100 |
| Capture | `PAD_NORMAL` / `PAD_BIKE` | `0.15` / `0.40` | crop padding |
| Capture | `DEDUP_SECONDS` | `2` | 0 - 120 |

Environment variables: `PORT` (default 5100), `NO_BROWSER=1`, `DCT_CONFIG` and `DCT_DATA` (use another
config file / data folder - handy for testing without touching your real setup).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Snapshot / Test says `Cannot reach host:port (timed out)` | The camera is offline, the port is wrong, or it is blocked. The tool checks the TCP port first so this fails in ~4 s instead of hanging. |
| `did not respond with a picture` | Port is open but the stream path or credentials are wrong, or the camera is at its connection limit. |
| Card says `reconnecting` | The stream dropped; it retries with backoff. The message under the chip says why. |
| Start says "Draw an ROI" | The camera has no ROI yet - use the ROI tab. |
| Start says "Model file not found" | Pick an existing model in Settings. |
| Settings Save says a class was ignored | The class name is not in the selected model; it is reported, never silent. |
| Nothing is captured | Check the ROI covers the road, confidence is not too high, classes include the vehicle type, and min size/overlap are not too strict. |
| Stop takes a few seconds | The stream was hung; shutdown waits for the reader (max ~10 s). |

## Development and tests

```bash
python -m pytest                          # ~100 s: unit + API + real-model integration (needs ../vedio and ../models)
python -m pytest --ignore=backend/tests/test_integration.py     # fast: ~10 s
python scripts/build_llms_full.py        # regenerate llms-full.txt after editing docs
```

Details for contributors and AI assistants: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
[`docs/API.md`](docs/API.md), [`AGENTS.md`](AGENTS.md).

## Credits and licenses

Detection by [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0 - check its terms for
your use). Icons: [Font Awesome Free 6.5.2](https://fontawesome.com) (icons CC BY 4.0, fonts SIL OFL 1.1,
code MIT), bundled locally in `frontend/static/vendor/fontawesome`. Font: DM Sans (SIL OFL 1.1) in
`frontend/static/vendor/fonts`. Both are local, so the UI works offline.
