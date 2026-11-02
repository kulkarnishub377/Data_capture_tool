# HTTP API

Base URL `http://127.0.0.1:5100`. JSON unless noted. Errors: `{"error": "message"}` with status 400/404/502.
Camera objects returned by the API have the RTSP password replaced by `****`.

## Cameras
| Method & path | Body / query | Result |
|---|---|---|
| `GET /api/cameras` | | list of cameras `{id,name,url(masked),enabled,rois,frame_size,device}` |
| `POST /api/cameras` | `{name,url,device?}` | the new camera (400 if name/url empty) |
| `PUT /api/cameras/<id>` | any of `{name,url,enabled,device}` | camera. `enabled:false` stops its worker. A masked URL (`****`) is ignored. |
| `DELETE /api/cameras/<id>` | | `{ok}` (stops its worker; images are kept) |
| `POST /api/cameras/<id>/test` | | `{ok:true,size:[w,h]}` or `{ok:false,error}` (<= ~15 s) |
| `GET /api/cameras/<id>/snapshot` | | JPEG with header `X-Frame-Size: WxH`; 502 + reason if unreachable |
| `PUT /api/cameras/<id>/roi` | `{rois:[[[x,y],...],...], frame_size:[w,h]}` (legacy `{points}` = one ROI) | camera. Every polygon needs >= 3 points; `[]` clears. |

## Settings, models, devices
| Method & path | Notes |
|---|---|
| `GET /api/settings` | all `config.ini` values, typed |
| `PUT /api/settings` | partial update. Validates choices/ranges (400 on error). If any worker-relevant key actually changed, running cameras restart once (counters kept). `MODEL` must exist in `MODELS_DIR`. |
| `GET /api/models` | sorted `*.pt` names in `MODELS_DIR` |
| `GET /api/model/classes?model=` | `{names, selected, default, warnings}`; `selected` honours `CLASSES`, unknown names go to `warnings` |
| `GET /api/devices` | `{gpus:[{id,name,memory_mb}], pool, warnings}` |

## Capture control
| Method & path | Notes |
|---|---|
| `POST /api/capture/<id>/start` | 400 if no ROI / model missing / model unloadable. `{ok,device,warnings}`; `{ok,already_running}` if running. Replaces a worker in `error`. |
| `POST /api/capture/<id>/stop` | `{ok}` |
| `GET /api/capture/status` | `{<id>: {state, message, saved, skipped, device}}`. `state`: `connected`, `reconnecting`, `restarting`, `stopping`, `stopped`, `error`. Counters include previous workers of the same camera. |
| `GET /api/stream/<id>` | MJPEG (`multipart/x-mixed-replace`) live preview with ROI + boxes; 404 if never started; survives restarts |

## Captured images
| Method & path | Notes |
|---|---|
| `GET /api/captures?camera=&date=&class=&page=&per_page=` | `{total,page,per_page,items:[{path,camera,date,kind,class,name,ts}]}`, newest first (`per_page` <= 200) |
| `GET /api/captures/<path>` | the JPEG. `path` is `items[].path`. |
| `DELETE /api/captures/<path>` | deletes the image and its `.json` |
| `GET /api/recent?limit=24` | `{items:[...]}` newest across cameras, one per vehicle (crop preferred). Fast (reads newest date folders only). |
| `GET /api/thumb/<path>?w=320` | cached JPEG thumbnail (`w` 32 - 1280), `Cache-Control: max-age=3600` |

`ts` is local time `YYYY-MM-DD HH:MM:SS`. Invalid / escaping paths return 400 or 404.
