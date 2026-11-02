# Architecture

A single Flask process serves a static frontend and a JSON API. Capture work runs in per-camera worker threads
inside that process. There is no database: state is JSON files, image files and `config.ini`.

```
Browser (frontend/)  <--HTTP/JSON, MJPEG-->  Flask app (backend/app.py)
                                               |-- CameraStore  -> data/cameras.json
                                               |-- settings     -> config.ini (edited in place)
                                               |-- storage      -> captures/**.jpg (+ .json), data/thumbs/
                                               '-- CameraWorker (one per running camera)
                                                     |-- LatestFrameReader thread  (RTSP -> newest frame)
                                                     |-- detect loop: YoloDetector + CaptureLogic
                                                     '-- saver thread (queue -> disk)
```

## Backend modules (`backend/`)

| File | Responsibility |
|---|---|
| `app.py` | `create_app(...)` builds the Flask app: REST API, MJPEG stream, restart-on-settings-change, status merging. `main()` runs it. Dependency-injected (`worker_factory`, `names_loader`, `device_detector`, `grabber`) so tests need no camera or GPU. |
| `capture_engine.py` | `CaptureLogic` (all per-frame decisions, thread-free and unit-tested), `YoloDetector`, `LatestFrameReader`, `CameraWorker` (threads, reconnect, preview JPEG), `default_classes`, `load_model_names`. |
| `dedup.py` | `Deduper`: track lock, IoU + perceptual-hash suppression limited to recently *active* saves; `iou`, `phash`. |
| `roi.py` | Polygon helpers: `point_in_polygon`, `box_overlap_ratio` (cached mask), `make_mask`, `expand_bbox`, `polygon_bbox`, `as_polygons`. Accept one polygon or a list. |
| `storage.py` | `CameraStore` (atomic JSON, migrates old single `roi` -> `rois`), `save_capture`, `list_captures`, `recent_captures`, `make_thumbnail`, `safe_capture_path`, `delete_capture`, `mask_url`. |
| `settings.py` | `load` / `save` for `config.ini`: typed values, defaults, choice + range validation, line-preserving writes. |
| `devices.py` | `detect` (cached GPU probe), `resolve` (parse `DEVICE`, fall back with warnings), `assign` (round-robin cameras over devices). |
| `camera_io.py` | `CameraError`, `check_port`, `run_with_timeout`, `open_capture` (5 s open/read timeouts), `grab_frame`. |

## Capture logic (`CaptureLogic.process`)

For every frame and every detection:
1. If the track is already locked (saved): `Deduper.touch` keeps it "active", then skip.
2. `eligible()`: class selected, conf >= `CONF_THRESH`, shorter side >= `MIN_BOX`, centre inside any ROI polygon,
   box overlap with the union mask >= `MIN_ROI_OVERLAP`.
3. `WHEN=first`: `_emit` immediately. `WHEN=best`: keep the best candidate (conf x area); finalize when the track
   is seen but no longer eligible (left the ROI) or unseen for `LOST_FRAMES` (10) frames; `flush` on stop.
4. `_emit` locks the track, takes the vehicle **crop**, rejects it if `Deduper.is_duplicate`, else records it.
   Returns capture requests; the saver thread writes them.

`Deduper` remembers a saved vehicle (box + hash) while it keeps being touched and for `DEDUP_SECONDS` after its
last sighting. IoU > 0.5 or pHash Hamming distance <= 6 inside that window means duplicate.

## Restart-on-save (`app.py`)

`PUT /api/settings` saves, compares old vs new for the worker-relevant keys, and if any changed calls
`restart_running()`: every running camera is marked in `restarting`, then a thread stops the old worker and
`start_worker` builds a new one. `start_worker` adds the old worker's counters to `carry[cid]` under the lock, so
`/api/capture/status` shows continuous counters; state reads `restarting` meanwhile. The MJPEG stream re-reads
`workers[cid]` each loop so the browser connection survives the swap. A failed restart is reported through `failed[cid]`.

## Frontend (`frontend/`)

Plain HTML/CSS/JS, no build step, no framework. `index.html` + `static/js/{app,live,roi,cameras,gallery,settings}.js`.
The Live tab toolbar (`live.js`) calls `/api/capture/start-all` and `/api/capture/stop-all` and stores the chosen cameras-per-row layout in `localStorage` (`liveCols`). `app.js` holds helpers (`h()` DOM builder, `api()`, `toast`, `showTab`, shared lightbox, `debounce`, `timeAgo`).
Each tab registers `tabs.<name>.onShow`. All DOM is built with `h()`/`textContent` (no `innerHTML`). Font Awesome
is vendored in `static/vendor/fontawesome`. Settings uses a *draft* object and sends it once on Save & restart.
Deep links: `/#settings`, `/#live`, ...

## Data formats

`data/cameras.json`:
```json
[{"id": "ab12cd34", "name": "Cam-01", "url": "rtsp://user:pass@host:554/s", "enabled": true,
  "rois": [[[x, y], [x, y], [x, y]]], "frame_size": [1920, 1080], "device": null}]
```
(`rois` = list of polygons, each >= 3 points in true frame pixels. The API never returns the real password.)

Capture file: `captures/<camera>/<YYYY-MM-DD>/<frame|crop>/<camera>_<class>_<YYYYmmdd_HHMMSS_mmm>[_N].jpg`.
Class names may contain underscores (`auto_rickshaw`); the camera comes from the folder name.

## Threading and safety notes

- `wlock` guards the `workers` dict; `CameraStore` has an `RLock` and writes via temp file + `os.replace`.
- `CameraWorker.stop()` sets the stop event, joins threads (<= 10 s each) and only then reports `stopped`.
- A worker exception sets state `error`, sets the stop event (so the saver exits) and `start` replaces it.
- Path traversal: all capture/thumbnail paths go through `storage.safe_capture_path` (rejects `..`, backslashes,
  absolute and drive paths; only `.jpg`).
- Flask debug mode is on by default but bound to 127.0.0.1 with the reloader off.
