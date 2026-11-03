# Architecture

A single Flask process serves a static frontend and a JSON API. Capture work runs in per-camera worker threads
inside that process. There is no database: state is JSON files, image files and `config.ini`.

```
Browser (frontend/)  <--HTTP/JSON, MJPEG-->  Flask app (backend/app.py)
                                               |-- CameraStore  -> data/cameras.json
                                               |-- settings     -> config.ini (edited in place)
                                               |-- storage      -> captures/**.jpg (+ .json); thumbnails in memory only
                                               '-- CameraWorker (one per running camera)
                                                     |-- LatestFrameReader thread  (RTSP -> newest frame)
                                                     |-- detect loop: YoloDetector + CaptureLogic
                                                     '-- saver thread (queue -> disk)
```

## Backend modules (`backend/`)

| File | Responsibility |
|---|---|
| `app.py` | `create_app(...)` builds the Flask app: REST API, MJPEG stream, restart-on-settings-change, status merging. `main()` runs it. Dependency-injected (`worker_factory`, `names_loader`, `device_detector`, `grabber`) so tests need no camera or GPU. |
| `capture_logic.py` | `CaptureLogic` + `Detection`: vehicle identity (links tracker-ID switches by motion prediction, drops double boxes), the best-frame **buffer**, when to save. Pure and unit-tested (no threads/model/camera). |
| `capture_engine.py` | `YoloDetector` (YOLO + ByteTrack via `trackers/bytetrack_vehicle.yaml`, class-agnostic NMS), `LatestFrameReader`, `CameraWorker` (threads, reconnect, preview JPEG), `default_classes`, `load_model_names`. |
| `dedup.py` | `iou`, `same_object` (one object boxed twice), `phash`. |
| `roi.py` | Polygon helpers: `point_in_polygon`, `box_overlap_ratio` (cached mask), `make_mask`, `expand_bbox`, `polygon_bbox`, `as_polygons`. Accept one polygon or a list. |
| `storage.py` | `CameraStore` (atomic JSON, migrates old single `roi` -> `rois`), `save_capture`, `list_captures`, `recent_captures`, `make_thumbnail` (in memory, reduced-size JPEG decode, never written to disk), `safe_capture_path`, `delete_capture`, `mask_url`. |
| `settings.py` | `load` / `save` for `config.ini`: typed values, defaults, choice + range validation, line-preserving writes. |
| `devices.py` | `detect` (cached GPU probe), `resolve` (parse `DEVICE`, fall back with warnings), `assign` (round-robin cameras over devices). |
| `model_diag.py` | `model_file_problem` (missing / empty / Git LFS pointer / truncated), `explain_model_error` (adds a plain-language hint to load errors), `MODEL_EXTS`. |
| `env_check.py` | `collect()` (torch/CUDA, TensorRT, OpenCV backends, ultralytics, lap) and `build_warnings()` (pure: turns the facts into fixes, e.g. a Jetson venv without `--system-site-packages`). |
| `torch_compat.py` | `apply()`: restores `torch.load(weights_only=False)` for our own trusted checkpoints (PyTorch >= 2.6); called before ultralytics is imported. |
| `camera_io.py` | `CameraError`, `check_port`, `run_with_timeout`; `open_stream` -> `OpenResult(cap, backend, note)`: RTSP via a GStreamer pipeline (`build_gst_pipeline`, `sanitize_rtsp_url`; hardware decode on a Jetson) with FFmpeg fallback, files/non-RTSP via FFmpeg; `grab_frame`. |
| `gpu_lock.py` | `guard(device)`: ONE process-wide lock for every GPU call (model load, inference, torch.cuda probes); a no-op for `cpu`. Required because a TensorRT engine load captures a CUDA graph. |

## Capture logic (`capture_logic.CaptureLogic.process`)

State: `_vehicles` (vid -> `_Vehicle`: last box/centre/size, smoothed velocity, `last_seen`, the buffered `best`
candidate, `saved`) and `_by_track` (tracker id -> vid).

For every frame:
1. `_suppress_overlaps`: detections of the same object (`dedup.same_object`: IoU > 0.5, or the smaller box mostly
   inside the larger and not tiny next to it) keep only the most confident one.
2. Detections whose tracker id is known go to their vehicle. A detection with an **unknown** id is matched to a
   vehicle that is **not detected in this frame** and was seen within `DEDUP_SECONDS`
   (`_find_lost_vehicle`): with two or more sightings the expected spot is last position + velocity x dt
   (tolerance `size x (1 + 1.5 dt)`, or it simply stopped); with one sighting the tolerance is `size x (1 + 3 dt)`.
   No match -> a new vehicle. Linked ids increment `merged` (shown as "skipped" in the UI).
3. For each vehicle: `_assess` -> eligible (class, conf, min size, centre inside an ROI, overlap with the ROI mask)
   / still inside. Eligible frames update the buffer when their `_score` beats the buffered one (the older frame
   is dropped). In `WHEN=first` the first eligible frame is saved at once.
4. `_finalize` (save once, release the buffer) when: the vehicle's centre leaves the ROIs, it is unseen for longer
   than `DEDUP_SECONDS`, it has been eligible for `MAX_HOLD_SECONDS`, or `flush()` (stop / restart).
5. Saved vehicles are kept (to absorb their own ID switches) and forgotten `DEDUP_SECONDS + 5 s` after last seen.

Requests returned: `{track_id, vehicle_id, cls, conf, box, frame, ts}`; the saver thread writes them.

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

- `wlock` guards all shared app state (`workers`, `carry`, `restarting`, `failed`); slow work (stopping a worker,
  loading a model) is never done while holding it. `names_lock` makes concurrent requests share one model load.
  `CameraStore` has an `RLock` and writes via temp file + `os.replace`.
- Inside a worker, `self._lock` guards everything other threads read (state, counters, backend, preview JPEG).
- `CameraWorker.stop()` sets the stop event, joins threads (<= 10 s each) and only then reports `stopped`.
- A worker exception flushes the vehicles already buffered (they are still saved), sets state `error`, sets the
  stop event (the saver writes what is queued, then exits) and `start` replaces it.
- The open stream is always released (`_loop` owns it). `LatestFrameReader.close()` never releases a capture while a
  `read()` is still running in the reader thread (that can crash FFmpeg/GStreamer); a blocked reader releases it itself.
- All GPU work, including the `.cpu()` copies of result tensors, runs inside `gpu_lock.guard(device)`.
- The preview JPEG is only encoded while a browser is watching (`latest_jpeg()` calls in the last 3 s). The MJPEG
  stream ends when the worker is `stopped` or `error`, so no server thread is held forever.
- Query numbers (`page`, `per_page`, `limit`, `w`) are clamped; a non-number gives 400, never 500.
- Path traversal: all capture/thumbnail paths go through `storage.safe_capture_path` (rejects `..`, backslashes,
  absolute and drive paths; only `.jpg`).
- The service binds `HOST` (default `0.0.0.0`) on `PORT` from `config.ini` and has no authentication: run it on a trusted network.
  `app.server_options()` keeps Flask's interactive debugger off whenever the host is not loopback (DEBUG logging stays on); the reloader is always off.
  No browser is opened at start; the banner prints the local and LAN URLs.
