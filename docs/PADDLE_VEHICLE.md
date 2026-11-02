# PP-Vehicle (PaddleDetection) vehicle detectors - evaluation

Source: <https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/deploy/pipeline/README.md> (PP-Vehicle model zoo).
Evaluated on 2026-11-02 with the project's sample video (`vedio/2_GetVIDESVideo.mp4`, 704x576) on a Windows PC, CPU only.
**Not yet run on the Jetson Orin NX** - use `scripts/paddle_vehicle_probe.py` there (below).

## The models

The README lists three vehicle detection / tracking archives (tracking uses the same archives as detection):

| Model | Archive | Size | README speed* |
|---|---|---|---|
| PP-YOLOE-L, high accuracy | `mot_ppyoloe_l_36e_ppvehicle.zip` | 182 MB | 25.7 ms |
| PP-YOLOE-S, lightweight | `mot_ppyoloe_s_36e_ppvehicle.zip` | 27 MB | 13.2 ms |
| PP-YOLOE+ t, ultra-light (320 px) | `ppyoloe_plus_crn_t_auxhead_320_60e_ppvehicle.tar.gz` | 17 MB | 10 ms (Jetson AGX) |

\* end-to-end pipeline times as printed on the page; the page does not say which GPU, except for the ultra-light one.
Download base URL: `https://bj.bcebos.com/v1/paddledet/models/pipeline/` (the L and S archives were downloaded and
verified: sizes match, zip test OK, contents read before use). The other PP-Vehicle features in the README (traffic flow
counting, wrong-way driving via PP-LiteSeg lane segmentation, plates, attributes) are separate pipelines, not detectors, and
are not needed for capturing training images.

What is inside each archive (L and S are identical in layout):
`model.pdmodel` + `model.pdiparams` (Paddle Inference format), `model.pdiparams.info`, `infer_cfg.yml`.
`infer_cfg.yml`: `arch: YOLO` (PP-YOLOE), **one class: `vehicle`**, resize to **640x640, keep_ratio false**, then Permute.

## Measured on your frames

Both models found the truck entering the road (frames 255-295) with scores 0.84-0.94 (L) and 0.65-0.94 (S) and drew correct
boxes. L also reported one weak 0.33 hit at frame 60 and, at frame 295, the small vehicle at the left edge (0.77); S missed
those. Input must be **raw 0-255 RGB** (dividing by 255 gives garbage: best score 0.06 vs 0.89). Inputs: `image`,
`scale_factor`; NMS is inside the model; output rows `[class, score, x1, y1, x2, y2]` in original-image pixels.
CPU time on this PC (oneDNN disabled, so not representative of a Jetson GPU): S about 0.8-1.0 s/frame, L about 2.5 s/frame.

## Compatibility with this tool

| | Verdict |
|---|---|
| Frame size / aspect ratio | OK. Any resolution; frames are resized to 640x640 without padding, exactly as the model was trained. |
| Detects vehicles on your video | OK for the truck on the sample. Not tested on auto-rickshaws, tractors, two-wheelers or night scenes. |
| Classes | **Limited.** One class, `vehicle`. No car / bus / truck / auto_rickshaw / bike. The tool's class filter and file names would all say `vehicle`. Your 7-class model is trained for your roads. |
| Model format | **Not drop-in.** The service runs Ultralytics (PyTorch / TensorRT) and cannot load `.pdmodel`. It needs a new Paddle detector backend. |
| Speed on the Orin NX | Unknown until measured on the device. |

Recommendation: if you want to evaluate them, take **S** (27 MB, about 2.5x faster than L, found the same truck at higher
confidence). Use L only if recall matters more than GPU time. Keep your trained model for class-based capture unless the
probe shows PP-Vehicle clearly finds vehicles yours misses.

## Set up on the Jetson (evaluation, no change to the service)

1. **Paddle with CUDA must exist on the device.** In the venv you will use:
   `python3 -c "import paddle; print(paddle.__version__, paddle.device.is_compiled_with_cuda())"`.
   If it is missing, install the NVIDIA-Jetson / JetPack 6 build of `paddlepaddle-gpu` (not the PyPI wheel; see your
   `deploy/ORIN_NX_SETUP.md` and Paddle's Jetson install guide). Do not install it into a venv shared with other services
   without checking.
2. **Get the models onto the device** (about 210 MB; `wget -c` resumes, the Baidu CDN can be slow):
   ```bash
   mkdir -p ~/_Data_capture_tool/models_paddle && cd ~/_Data_capture_tool/models_paddle
   wget -c https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_ppvehicle.zip
   wget -c https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip
   unzip mot_ppyoloe_s_36e_ppvehicle.zip && unzip mot_ppyoloe_l_36e_ppvehicle.zip
   ```
   (Or copy `D:\model_train\models\paddle_ppvehicle\` from the PC.) Keep them out of the `models/` folder the tool scans;
   it only lists `.pt/.engine/.onnx` files and would ignore the folders anyway.
3. **Run the probe on your frames** (a video file, or an RTSP URL, which uses the service's GStreamer/FFmpeg opener):
   ```bash
   cd ~/_Data_capture_tool
   python3 scripts/paddle_vehicle_probe.py --model models_paddle/mot_ppyoloe_s_36e_ppvehicle \
       --source "rtsp://user:pass@host:554/stream" --device gpu --every 5 --frames 30
   python3 scripts/paddle_vehicle_probe.py --model models_paddle/mot_ppyoloe_s_36e_ppvehicle \
       --source video.mp4 --device gpu --trt fp16       # Paddle-TensorRT; the first call builds the engine (slow)
   ```
   It prints detections per frame, the median ms per frame (excluding the first call) and writes annotated images to
   `paddle_probe_out/`. On CPU, oneDNN is disabled by default because models exported with Paddle 2.x break in Paddle 3's
   oneDNN passes (`--mkldnn` re-enables it).
4. **Compare** with your Ultralytics `.engine` on the same stream: how many vehicles each finds, false positives, ms per
   frame.

## If the results justify it: integrating into the tool (not built)

A `PaddleDetector` backend next to `YoloDetector` (same `Detection` output, class `vehicle`), with ByteTrack from
Ultralytics supplying track IDs, behind the same GPU lock, selectable in Settings. Open questions to settle on the device
first: Paddle 3.x loading these Paddle-2.x-exported models on the GPU (the CPU oneDNN failure shows version sensitivity),
Paddle and PyTorch sharing the Orin's 16 GB, and whether Paddle-TensorRT reaches the frame rate you need. The alternative
is converting to ONNX (`paddle2onnx`) and building a TensorRT engine on the device, which also needs custom box decoding
because it is not an Ultralytics model.

Licence note: PaddleDetection is Apache-2.0. I did not check the licences of the datasets the weights were trained on;
check before commercial use.
