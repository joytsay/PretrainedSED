# C++ TensorRT aggregate SED testbed (Jetson AGX Orin)

The C++ implementation consists of two executables:

- `atst_sed_worker` is a persistent callback process. It receives timestamped
  40 ms mono PCM packets, maintains a silence-prefilled 10-second rolling
  buffer per stream, performs TensorRT inference, and writes JSON callbacks to
  standard output.
- `atst_sed_testbed` is a Qt 6 frontend. It provides a multi-select media file
  picker and playlist, extracts 16 kHz mono PCM through FFmpeg in real time,
  supplies `cam_id` and media timestamps to the worker, plays audio/video, and
  dynamically displays every aggregate class loaded by the worker. A class and
  percentage turn red above the selected threshold.

All C++ sources are in this directory. The TensorRT engine consumes normalized
ATST mel tensors shaped `[1,1,64,1001]` and returns `[1,447,250]` source
probabilities. At startup, the worker reads `class_mapping.csv` and the ordered
`ATST-F_strong_1.labels.txt` vocabulary produced by the converter. Every unique
`class_name` in `class_mapping.csv` becomes an output class. The worker sums
each class's configured sources and caps the result at 100%.

## Convert the model

Build on AGX with:

docker build --no-cache --pull \
  --platform linux/arm64 \
  -f docker/agx.Dockerfile \
  -t psed:latest \
  .

On the AGX container, convert the public checkpoint to ONNX and TensorRT FP16:

```sh
python scripts/export_atst_f_tensorrt.py \
  --checkpoint /workspace/resources/ATST-F_strong_1.pt \
  --output /workspace/resources/ATST-F_strong_1.trt
```

The converter also creates
`/workspace/resources/ATST-F_strong_1.labels.txt` for the worker.

To retain or choose the intermediate ONNX path, pass `--onnx PATH`. Use
`--onnx-only` to stop before TensorRT, `--fp32` for an FP32 engine, or
`--trtexec /usr/src/tensorrt/bin/trtexec` when `trtexec` is not on `PATH`.
If `trtexec` is unavailable, the converter automatically uses the Python
`tensorrt` Builder API. After a successful ONNX export, `--engine-only` reuses
the existing `.onnx` and `.labels.txt` files without exporting them again.
TensorRT engines are hardware/TensorRT-version specific, so build the `.trt`
file on the target AGX rather than copying one built on an x86 workstation.

## Build

```sh
cmake -S /workspace -B /workspace/build-agx \
  -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build /workspace/build-agx -j4
```

## Run the testbed

```sh
/workspace/build-agx/atst_sed_testbed \
  --worker /workspace/build-agx/atst_sed_worker \
  --engine /workspace/resources/ATST-F_strong_1.trt \
  --mapping /workspace/class_mapping.csv \
  --labels /workspace/resources/ATST-F_strong_1.labels.txt \
  --ffmpeg /usr/bin/ffmpeg \
  --cam-id camera-01
```

The Qt process needs access to the AGX display and audio service. When launched
inside Docker, pass through the appropriate X11/Wayland socket and display
environment for the desktop session.

## Worker callback protocol

The worker can also be integrated without the frontend. It first emits a
`ready` callback; input messages and callbacks use newline-delimited JSON.
Audio is mono signed 16-bit little-endian PCM encoded as base64. Every packet
must contain exactly 640 samples (1280 decoded bytes), which is 40 ms at
16 kHz:

```text
callback: {"event":"ready","classes":["Glass breaking","Gunshot","Siren"],"sample_rate":16000,"channels":1,"encoding":"s16le","packet_samples":640,"packet_ms":40,"window_ms":10000,"silence_prefill_ms":10000}
request:  {"type":"stream_start","id":1,"cam_id":"camera-01","timestamp_ms":0}
request:  {"type":"audio","id":1,"cam_id":"camera-01","timestamp_ms":0,"sample_rate":16000,"channels":1,"encoding":"s16le","audio_b64":"..."}
callback: {"event":"result","id":1,"cam_id":"camera-01","timestamp_ms":0,"window_start_ms":-9960,"window_end_ms":40,"processing_ms":31,"superseded_packets":0,"scores":[0.01,0.02,0.03]}
request:  {"type":"audio","id":1,"cam_id":"camera-01","timestamp_ms":40,"sample_rate":16000,"channels":1,"encoding":"s16le","audio_b64":"..."}
request:  {"type":"stream_end","id":1,"cam_id":"camera-01","timestamp_ms":10000}
callback: {"event":"complete","id":1,"cam_id":"camera-01","packets_received":250,"last_timestamp_ms":9960}
```

At `stream_start`, the worker fills the 10-second buffer with silence. The first
audio packet drops the oldest 640 silent samples, inserts the received samples,
and immediately triggers inference. After 250 packets, the window contains only
received audio. Audio reception and inference run independently. If inference
takes longer than 40 ms, a newly received window replaces the older pending
window instead of queuing stale inference work. `superseded_packets` reports
how many pending windows that result replaced. Consequently, result timestamps
can skip under load, but they remain close to the current media timestamp rather
than drifting progressively behind it. The result timestamp is the start time
of the newest 40 ms output frame. Negative initial `window_start_ms` values
denote the silent history before the stream began.

The `classes` array in the `ready` callback and the `scores` array in each
result always have the same dynamic length and matching order. Class order is
the order in which each unique `class_name` first appears in
`class_mapping.csv`.

The Qt confidence panel sorts its visible rows by current confidence, with the
highest score at the top. This affects only presentation; the worker's
`classes` and `scores` arrays retain their stable CSV-defined order.

The Qt testbed starts FFmpeg first and holds media playback at the requested
position until the first result arrives. This one-inference pre-roll aligns the
subsequent result timestamps with the video timeline. Pause and seek operations
also pause or restart the audio stream so the two timelines do not drift apart.
