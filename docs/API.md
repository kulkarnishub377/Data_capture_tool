# HTTP API

Base URL `http://<host>:<PORT>` (default port 5100; the service listens on `0.0.0.0`, there is no authentication). JSON unless noted. Errors: `{"error": "message"}` with status 400/404/502.
Camera objects returned by the API have the RTSP password replaced by `****`.

## Cameras
| Method & path | Body / query | Result |
|---|---|---|
| `GET /api/cameras` | | list of cameras `{id,name,url(masked),enabled,rois,frame_size,device}` |
| `POST /api/cameras` | `{name,url,device?}` | the new camera (400 if name/url empty, or the URL is neither an rtsp/rtsps/http/https/rtmp URL with a host nor an existing video file) |
| `PUT /api/cameras/<id>` | any of `{name,url,enabled,device}` | camera. `enabled:false` stops its worker. A masked URL (`****`) is ignored; an invalid URL gives 400. |
| `DELETE /api/cameras/<id>` | | `{ok}` (stops its worker; images are kept) |
| `POST /api/cameras/<id>/test` | | `{ok:true,size:[w,h]}` or `{ok:false,error}` (<= ~15 s) |
| `GET /api/cameras/<id>/snapshot` | | JPEG with header `X-Frame-Size: WxH`; 502 + reason if unreachable |
| `PUT /api/cameras/<id>/roi` | `{rois:[[[x,y],...],...], frame_size:[w,h]}` (legacy `{points}` = one ROI) | camera. Every polygon needs >= 3 points; `[]` clears. |

## Settings, models, devices
| Method & path | Notes |
|---|---|
| `GET /api/settings` | all `config.ini` values, typed |
| `PUT /api/settings` | partial update. Validates choices/ranges (400 on error). If any worker-relevant key actually changed, running cameras restart once (counters kept). `MODEL` must exist in `MODELS_DIR`. |
| `GET /api/models` | sorted model files in `MODELS_DIR`: `*.pt`, `*.engine` (TensorRT), `*.onnx` |
| `POST /api/model/test?model=` | loads the model and runs one blank test frame on the configured device. Always 200: `{ok:true, model, device, classes, seconds, message}` or `{ok:false, model, device?, error}` where `error` has a plain-language hint |
| `GET /api/environment` | what this Python can use: `{python, version, machine, in_venv, system_site_packages, torch:{version,cuda,gpu}, tensorrt, opencv:{version,gstreamer,ffmpeg}, nvdec, ultralytics, lap, warnings:[...]}` |
| `GET /api/paths` | `{models_dir, models_dir_exists, models_count, output_dir}`: the resolved folders (`~`, env vars and relative paths expanded) |
| `GET /api/model/classes?model=` | `{names, selected, default, warnings}`; `selected` honours `CLASSES`, unknown names go to `warnings` |
| `GET /api/devices` | `{gpus:[{id,name,memory_mb}], pool, warnings}` |

## Capture control
| Method & path | Notes |
|---|---|
| `POST /api/capture/<id>/start` | 400 if no ROI / model missing / model unloadable. `{ok,device,warnings}`; `{ok,already_running}` if running. Replaces a worker in `error`. |
| `POST /api/capture/<id>/stop` | `{ok}` |
| `POST /api/capture/start-all` | Starts every **enabled** camera. `{started, failed, skipped, already_running, results:{<id>:{name, ok\|error\|skipped\|already_running}}}`. A camera that cannot start (no ROI, model missing) is reported, never fatal. |
| `POST /api/capture/stop-all` | Stops all running cameras in parallel. `{stopped: n}` |
| `GET /api/capture/status` | `{<id>: {state, message, saved, skipped, device, backend}}` (`backend`: `gstreamer` / `ffmpeg` / `file` once a stream is open). `state`: `connected`, `reconnecting`, `restarting`, `stopping`, `stopped`, `error`. Counters include previous workers of the same camera. |
| `GET /api/stream/<id>` | MJPEG (`multipart/x-mixed-replace`) live preview with ROI + boxes; 404 if never started; survives restarts |

## Captured images
| Method & path | Notes |
|---|---|
| `GET /api/captures?camera=&date=&class=&kind=&sort=&q=&page=&per_page=` | `{total,page,per_page,items:[{path,camera,date,kind,class,name,ts}],categories:[...],cameras:[...],dates:[...]}`. `sort` = `newest` (default) or `oldest`; `kind` = `frame` / `crop`; `class` = one or comma-separated classes; `q` = case-insensitive search of name, camera, class and time over all pages; `camera` = a camera folder name from `cameras`. `categories`/`cameras`/`dates` list everything on disk (for the filter controls). `per_page` <= 200; a non-number gives 400. |
| `GET /api/captures/<path>` | the JPEG. `path` is `items[].path`. |
| `DELETE /api/captures/<path>` | deletes the image and its `.json` |
| `POST /api/captures/delete-batch` | `{paths:[...]}` (at most 5000) -> `{ok:true,deleted:n,errors:["path: reason", ...]}`. Deletes each image and its `.json`; a bad or missing entry is reported in `errors` and never stops the rest. |
| `GET /api/recent?limit=10` | `{items:[...]}` newest across cameras, one per vehicle (crop preferred); default and UI use the latest 10 (max 100). Fast (reads newest date folders only). |
| `GET /api/thumb/<path>?w=320` | JPEG thumbnail (`w` 32 - 1280) made in memory for each request - nothing is stored on the server (the browser may cache it: `Cache-Control: private, max-age=3600`); 404 if the image is gone |

`ts` is local time `YYYY-MM-DD HH:MM:SS`. Invalid / escaping paths return 400 or 404.
