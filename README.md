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

The service listens on all network interfaces (`HOST = 0.0.0.0`) on the port set in `config.ini` (default **5100**).
It does **not** open a browser. At start it prints the addresses to use:

```
Data Capture Tool listening on 0.0.0.0:5100
  this machine : http://127.0.0.1:5100
  other machines: http://<this-pc-ip>:5100
```

Open one of those in a browser (other machines use the second address; allow the port in the firewall). Then:

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
| **Live** | Settings summary; a camera toolbar with **Start all / Stop all** and a **Layout** selector (Auto, or 1-4 cameras per row, remembered); one live card per camera (video with ROI and detection boxes, status, counters, **full-screen** button, Start/Stop); and a **Recent captures** panel (the latest 10) with thumbnails + class tags. Click a thumbnail to enlarge or delete. |
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
| | Capture moment | **Best frame (buffered)** (default): the best view of each vehicle is held in a buffer, a better frame replaces the older one, and it is saved once when the vehicle leaves the ROI. **First frame (instant)** saves immediately (lower quality, more duplicates). |
| | Keep metadata | Off (default) = images only. On = a `.json` next to each image. |
| Advanced | Minimum vehicle size | Shorter side in px (default 40). |
| | Minimum overlap with ROI | 0 - 1 (default 0.5). Stops half-in vehicles being saved. |
| | Duplicate window | Seconds (default 2). If the tracker loses a vehicle and gives it a new ID, a track that appears where it was expected within this time is the **same** vehicle. |
| | Maximum wait | Seconds (default 10). A vehicle that stays in the ROI (stopped / traffic) is saved after this long. |
| | JPEG quality | 1 - 100 (default 95). |
| | Detection size | Model input size in px (default 640). |

---

## How capture works

Per camera, three threads run:

1. **Reader** - reads the RTSP stream (TCP) as fast as it arrives and keeps only the **newest frame**, so slow
   inference never makes the picture lag. Reconnects automatically; open/read timeouts are 5 s.
2. **Detect + track** - runs YOLO with ByteTrack (settings in `backend/trackers/bytetrack_vehicle.yaml`, class-agnostic
   NMS) on the ROI area plus a 30% margin, so a vehicle half outside the ROI is seen whole.
3. **Saver** - writes files from a queue so a slow disk never drops frames.

Each camera loads **its own model instance** (ByteTrack state is per instance) on its assigned device.

### One capture per *vehicle* (not per tracker ID)

A tracker gives a vehicle an ID, but IDs switch (a missed detection, occlusion, a fast vehicle at a low frame
rate) and then one vehicle looks like two. The capture logic (`backend/capture_logic.py`) therefore keeps its own
notion of a **vehicle**:

- **Same object boxed twice** in one frame (e.g. 46x40 and 53x65 on the same car, or as "car" and "truck") is
  one vehicle: the weaker box is dropped.
- **A new tracker ID that appears where a just-lost vehicle was expected** (its last position + its speed, with a
  tolerance that grows over the *duplicate window*) is the **same vehicle**. A vehicle that is still being
  detected never absorbs another track, so two cars driving close together stay two vehicles.
- A vehicle that has been saved keeps absorbing its own ID switches, so it is never saved a second time.

### The buffer: keep the best frame, drop the older ones

While a vehicle is inside an ROI its frames are **not saved**. Only the best candidate is held in memory
(score = confidence x size x how much of it is inside the ROI, penalised if cut by the frame edge); when a
better frame arrives it **replaces** the buffered one and the old frame is dropped. The vehicle is saved **once**:

| When | Why |
|---|---|
| it leaves the ROI | the normal case |
| it has not been seen for the duplicate window | it is gone for good |
| it has stayed `MAX_HOLD_SECONDS` in the ROI | stopped vehicles / traffic jams |
| you stop capture or restart for new settings | nothing buffered is lost (flush) |

`First frame (instant)` mode skips the buffer and saves the first good frame (the vehicle is still remembered, so
ID switches do not duplicate it).

The next car in the same lane is captured normally: a different car arriving later, or entering behind the
previous one, is not mistaken for it. File names never collide (`_2`, `_3` suffix).

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
| App | `DEBUG` | `true` | DEBUG logging + Flask debug mode (reloader off). The interactive debugger is **never** enabled when the server is reachable from the network. |
| App | `HOST` | `0.0.0.0` | `0.0.0.0` = reachable from other machines; `127.0.0.1` = this machine only. |
| App | `PORT` | `5100` | Web UI port (1 - 65535). Edit it here and restart the service. |
| Paths | `MODELS_DIR` | `../models` | Relative to `config.ini`. `~` = the home folder of the user running the service (e.g. `~/_VIDES/_backend_service_combined/models`); environment variables are expanded; a path starting with `/` is taken from the filesystem root. |
| Paths | `OUTPUT_DIR` | `./captures` | |
| Model | `MODEL` | | file inside `MODELS_DIR` |
| Model | `DEVICE` | `auto` | `auto`, `cpu`, `cuda:0`, `cuda:0,cuda:1` |
| Model | `HALF` | `true` | FP16 on CUDA only |
| Model | `CONF_THRESH` | `0.25` | 0.01 - 1 |
| Model | `IMG_SIZE` | `640` | 160 - 2048 |
| Model | `CLASSES` | empty | class **names**, comma separated; empty = auto |
| Capture | `SAVE_MODE` | `frame` | `frame` / `crop` / `both` |
| Capture | `SAVE_METADATA` | `false` | |
| Capture | `WHEN` | `best` | `best` (buffer the best frame, save once) / `first` (save immediately) |
| Capture | `MIN_BOX` | `40` | 1 - 5000 |
| Capture | `MIN_ROI_OVERLAP` | `0.5` | 0 - 1 |
| Capture | `JPEG_QUALITY` | `95` | 1 - 100 |
| Capture | `PAD_NORMAL` / `PAD_BIKE` | `0.15` / `0.40` | crop padding |
| Capture | `DEDUP_SECONDS` | `2` | 0 - 120. Window for re-linking a lost vehicle after a tracker ID switch |
| Capture | `MAX_HOLD_SECONDS` | `10` | 1 - 600. Save a vehicle that stays inside the ROI this long |

Environment variables: `HOST` and `PORT` (override `config.ini`), `DCT_CONFIG` and `DCT_DATA` (use another
config file / data folder - handy for testing without touching your real setup).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Other machines cannot open the page | Check `HOST = 0.0.0.0`, that you use the address printed at start, and that the OS firewall allows the `PORT` (Windows: inbound rule for TCP 5100). |
| Model does not load / Start says "Could not load model" | Settings > Model: press **Test model** (exact error + hint) and **Check environment**. Typical causes: the venv cannot see the system CUDA torch / TensorRT (`--system-site-packages`), a `.engine` built on another device, a Git LFS pointer or truncated file instead of the real model, an older Ultralytics than the one that trained the model. |
| No models in the dropdown | `MODELS_DIR` does not exist for the user running the service, or has no `*.pt` files. Settings > Model shows the resolved folder and whether it was found (`GET /api/paths`). |
| Snapshot / Test says `Cannot reach host:port (timed out)` | The camera is offline, the port is wrong, or it is blocked. The tool checks the TCP port first so this fails in ~4 s instead of hanging. |
| `did not respond with a picture` | Port is open but the stream path or credentials are wrong, or the camera is at its connection limit. |
| Card says `reconnecting` | The stream dropped; it retries with backoff. The message under the chip says why. |
| Start says "Draw an ROI" | The camera has no ROI yet - use the ROI tab. |
| Start says "Model file not found" | Pick an existing model in Settings. |
| Settings Save says a class was ignored | The class name is not in the selected model; it is reported, never silent. |
| One vehicle captured twice | Raise **Duplicate window** a little (e.g. 3-4 s) or use a faster model/device so the tracker keeps up; check `Best frame` mode is on. Two vehicles close together are deliberately kept separate. |
| Nothing is captured | Check the ROI covers the road, confidence is not too high, classes include the vehicle type, and min size/overlap are not too strict. |
| Stop takes a few seconds | The stream was hung; shutdown waits for the reader (max ~10 s). |

## Running on a Linux / Jetson device

```bash
bash start.sh                  # uses ./venv if present, else python3  (PYTHON=/path/to/python bash start.sh to choose)
```

On an NVIDIA Jetson, CUDA PyTorch, TensorRT and the GStreamer OpenCV normally come from JetPack as **system**
packages. Install so that pip does not hide them (see `requirements-jetson.txt`):

```bash
python3 -m venv --system-site-packages venv && source venv/bin/activate
pip install --no-deps ultralytics && pip install -r requirements-jetson.txt
```

Do not `pip install torch`, `torchvision`, `tensorrt` or `opencv-python` there. A venv created *without*
`--system-site-packages`, or with pip's own torch/OpenCV, cannot see the GPU or TensorRT and models will not load.
You can also reuse an existing venv on the device that already runs your other AI services.

**Do not guess - ask the device.** In *Settings > Model & device*:
- **Check environment** shows what this Python can use (PyTorch + CUDA, TensorRT, OpenCV with GStreamer/FFmpeg,
  Ultralytics, the `lap` tracker module) and prints a plain-language fix for each problem.
- **Test model** loads the selected model and runs one test frame on the configured device, then reports success or
  the exact error with a hint (Git LFS pointer, truncated file, wrong Ultralytics version, missing CUDA/TensorRT...).

Model files: `*.pt` (PyTorch), `*.engine` (TensorRT, much faster on a Jetson; must be built **on that device**) and
`*.onnx` are all listed. `.engine` files need a CUDA GPU (`DEVICE = auto` or `cuda:0`), never `cpu`.

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
