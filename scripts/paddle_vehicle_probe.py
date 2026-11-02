"""Try a PP-Vehicle (PaddleDetection PP-YOLOE) vehicle detector on your own frames, standalone.

This is an evaluation tool, not part of the capture service: the service runs Ultralytics models, and Paddle models
(model.pdmodel + model.pdiparams) cannot be loaded by it. Use this to see what a Paddle model detects on YOUR cameras and how
fast it is on THIS device before deciding whether it is worth integrating.

  python scripts/paddle_vehicle_probe.py --model models/paddle_ppvehicle/mot_ppyoloe_s_36e_ppvehicle --source video.mp4
  python scripts/paddle_vehicle_probe.py --model ... --source "rtsp://user:pass@host:554/stream" --device gpu --trt fp16

What the model expects (from its infer_cfg.yml): RGB image resized to 640x640 (no aspect-ratio padding), RAW 0-255 float32
values (do NOT divide by 255), CHW; inputs 'image' and 'scale_factor'. NMS is inside the model. Output rows:
[class_id, score, x1, y1, x2, y2] in original-image pixels. The only class is 'vehicle'.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

INPUT_SIZE = 640


def prepare_image(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR frame -> float32 [1, 3, 640, 640], RGB, raw 0-255 (the model normalises internally)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC).astype("float32")
    return img.transpose(2, 0, 1)[None]


def scale_factor(frame_bgr: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    return np.array([[INPUT_SIZE / h, INPUT_SIZE / w]], dtype="float32")


def parse_detections(rows: np.ndarray, conf: float) -> list:
    """Model output rows -> [{'score', 'box': (x1, y1, x2, y2)}] above `conf`, best first."""
    out = []
    if rows is not None and getattr(rows, "ndim", 0) == 2:
        for cls, score, x1, y1, x2, y2 in rows[:, :6]:
            if cls >= 0 and score >= conf:
                out.append({"score": float(score), "box": (int(x1), int(y1), int(x2), int(y2))})
    return sorted(out, key=lambda d: -d["score"])


def load_predictor(model_dir: Path, device: str, trt: str, mkldnn: bool):
    from paddle.inference import Config, PrecisionType, create_predictor
    cfg = Config(str(model_dir / "model.pdmodel"), str(model_dir / "model.pdiparams"))
    if device == "gpu":
        cfg.enable_use_gpu(500, 0)
        if trt:
            cfg.enable_tensorrt_engine(workspace_size=1 << 30, max_batch_size=1, min_subgraph_size=3,
                                       precision_mode=PrecisionType.Half if trt == "fp16" else PrecisionType.Float32,
                                       use_static=False, use_calib_mode=False)
    else:
        cfg.disable_gpu()
        cfg.set_cpu_math_library_num_threads(4)
        if not mkldnn:
            cfg.disable_mkldnn()          # models exported with Paddle 2.x break in Paddle 3's oneDNN passes
    cfg.enable_memory_optim()
    cfg.disable_glog_info()
    return create_predictor(cfg)


def detect(predictor, frame_bgr, conf):
    feeds = {"image": prepare_image(frame_bgr), "scale_factor": scale_factor(frame_bgr)}
    for name in predictor.get_input_names():
        predictor.get_input_handle(name).copy_from_cpu(feeds.get(name, np.array([[INPUT_SIZE, INPUT_SIZE]], "float32")))
    predictor.run()
    rows = predictor.get_output_handle(predictor.get_output_names()[0]).copy_to_cpu()
    return parse_detections(rows, conf)


def frames_from(source: str, every: int, limit: int):
    """Frames from a video file or an RTSP URL (RTSP uses the service's own GStreamer/FFmpeg opener)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    import camera_io
    res = camera_io.open_stream(source, is_file=Path(source).is_file())
    if res.cap is None:
        raise SystemExit(f"cannot open {source}: {res.note}")
    print(f"opened with {res.backend}")
    n = taken = 0
    while taken < limit:
        ok, frame = res.cap.read()
        if not ok:
            break
        if n % every == 0:
            taken += 1
            yield n, frame
        n += 1
    res.cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="folder with model.pdmodel / model.pdiparams / infer_cfg.yml")
    ap.add_argument("--source", required=True, help="video file or RTSP URL")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--trt", choices=["", "fp16", "fp32"], default="", help="Paddle-TensorRT (gpu only)")
    ap.add_argument("--mkldnn", action="store_true", help="enable oneDNN on CPU (can fail on models exported by Paddle 2.x)")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--every", type=int, default=5, help="use every Nth frame")
    ap.add_argument("--frames", type=int, default=30, help="how many frames to test")
    ap.add_argument("--out", default="paddle_probe_out", help="folder for annotated images")
    a = ap.parse_args()

    model_dir = Path(a.model)
    predictor = load_predictor(model_dir, a.device, a.trt, a.mkldnn)
    print(f"model {model_dir.name} | device {a.device}{' + TensorRT ' + a.trt if a.trt else ''}")
    print("inputs :", predictor.get_input_names(), "| outputs:", predictor.get_output_names())
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    times, total, with_hits, saved = [], 0, 0, 0
    for idx, frame in frames_from(a.source, a.every, a.frames):
        t0 = time.perf_counter()
        dets = detect(predictor, frame, a.conf)
        dt = (time.perf_counter() - t0) * 1000
        if len(times) or idx > 0:       # the very first call includes graph/engine build: report but do not average it
            times.append(dt)
        total += len(dets)
        with_hits += bool(dets)
        print(f"frame {idx:5}: {len(dets)} vehicle(s)  {dt:6.0f} ms  " +
              " ".join(f"[{d['box'][0]},{d['box'][1]},{d['box'][2]},{d['box'][3]} s={d['score']:.2f}]" for d in dets[:3]))
        if dets and saved < 5:
            img = frame.copy()
            for d in dets:
                x1, y1, x2, y2 = d["box"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"vehicle {d['score']:.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imwrite(str(out_dir / f"{model_dir.name}_{idx}.jpg"), img)
            saved += 1
    if times:
        print(f"\n{with_hits} frames with detections, {total} detections total")
        print(f"speed: median {statistics.median(times):.0f} ms, mean {statistics.mean(times):.0f} ms per frame "
              f"(~{1000 / statistics.median(times):.1f} fps) - detector only, excluding the first call")
    print(f"annotated images in {out_dir}/")


if __name__ == "__main__":
    main()
