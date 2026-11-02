# AGENTS.md - guide for AI coding assistants

Data Capture Tool: a Flask + plain-JS app that captures one training image per vehicle from RTSP streams inside
user-drawn ROIs. Read `README.md` (usage), `docs/ARCHITECTURE.md` (design) and `docs/API.md` (endpoints) first.
A compact index is in `llms.txt`; the whole documentation in one file is `llms-full.txt`.

## Layout
- `backend/` Python: `app.py` (API), `capture_engine.py` (workers, detector), `capture_logic.py` (vehicle identity + buffer), `dedup.py`, `roi.py`,
  `storage.py`, `settings.py`, `devices.py`, `camera_io.py`, `tests/`.
- `frontend/` static UI: `index.html`, `static/js/*.js`, `static/css/app.css`, `static/vendor/fontawesome/`.
- `config.ini` (settings), `data/` (cameras.json, thumbs/), `captures/` (output). `data/`, `captures/`, models are git-ignored.

## Run and test
```bash
python backend/app.py                                  # HOST/PORT from config.ini [App] (env HOST, PORT override; DCT_CONFIG, DCT_DATA)
python -m pytest --ignore=backend/tests/test_integration.py     # fast suite (~10 s)
python -m pytest                                       # + real-model integration (~100 s; needs ../vedio and ../models)
python scripts/build_llms_full.py                      # regenerate llms-full.txt after editing README/docs/AGENTS
```
Use `DCT_CONFIG` / `DCT_DATA` to run against a throw-away config and data folder; never test against the user's
real `config.ini` or `data/cameras.json` (it holds a real camera URL with a password).

## Conventions that matter
- **TDD**: write the failing test first (`backend/tests/`), watch it fail, then implement. `CaptureLogic`,
  `Deduper`, `storage`, `settings`, `devices` are pure and unit-tested; the API is tested with a fake worker
  (`worker_factory`) so no camera/GPU is needed. Keep that dependency injection.
- **No database.** State = `data/cameras.json` (atomic write), `config.ini`, image files. Do not add a DB.
- **`settings.save` edits lines in place** (comments survive). Add new keys to `DEFAULTS` (+ `CHOICES`/`RANGES`)
  and to `WORKER_KEYS` in `app.py` if a running worker must restart when it changes.
- **ROI data model** is `rois` (list of polygons). `roi.py` functions accept one polygon or a list.
- **Model loading failures must explain themselves**: go through `model_diag` (`ModelFileError`, `explain_model_error`) and surface the text in
  the UI (Settings > Test model / Check environment). The deployment target is a Jetson whose setup you cannot see - never assume it.
- **Never expose camera passwords**: API responses go through `public()` / `storage.mask_url`.
- **Paths from clients** must go through `storage.safe_capture_path`. **Configured folders** (`MODELS_DIR`, `OUTPUT_DIR`) go through
  `settings.resolve_path` (`~`, env vars, relative-to-config, absolute) - never `base / value` directly.
- **Frontend**: no framework, no build step, no `innerHTML`; build DOM with `h()`. Settings edits are a *draft*
  applied only by "Save & restart". Icons are Font Awesome (vendored). Colours come only from the CSS variables at the top of `static/css/app.css` (dark default = VIDES palette, light via `[data-theme=light]`); never hard-code UI colours in JS/HTML (the ROI polygon colours in `roi.js` are intentional annotation colours). The page has no JS test runner here;
  UI changes were verified by driving headless Chrome over the DevTools protocol.
- **Capture logic** (`capture_logic.py`) is per *vehicle*, not per tracker id: it links ID switches by motion prediction, buffers
  the best frame and saves once. A vehicle that is still detected must never absorb a new track (two close cars stay two),
  and a different car arriving later must not be absorbed (it would be lost). Both have regression tests - keep them.
- **Network exposure**: the service binds `0.0.0.0` by default and has no login. `server_options()` must keep the Werkzeug
  interactive debugger OFF unless the host is loopback (it is remote code execution). It never opens a browser.
- Windows is the primary platform: use `pathlib`, avoid shell-specific code, `taskkill` not `kill`.

## Gotchas
- Worker states: `connected`, `reconnecting`, `restarting`, `stopping`, `stopped`, `error`. `error` workers are
  replaced on Start. `restarting` is synthesised in `/api/capture/status` while a settings restart is in flight.
- On Windows a second server can bind a port that an older one still holds - kill stale test servers
  (find them by `PORT`/`DCT_CONFIG` env) before re-testing.
- `cv2.VideoCapture` on an unreachable host hangs ~30 s: always go through `camera_io` (port check + timeouts).
- COCO models are recognised by class names (`person` + the four vehicle classes); anything else is "custom".
- Saved image names: `<cam>_<class>_<YYYYmmdd_HHMMSS_mmm>[_N].jpg`; class names can contain underscores.

## Do not
- Commit `config.ini` secrets, `data/`, `captures/`, `*.pt`. Do not start capture automatically.
- Add noisy dependencies; keep `requirements.txt` small.
