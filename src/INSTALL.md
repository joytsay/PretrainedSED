# C++ TensorRT aggregate SED testbed (Jetson AGX Orin)

The C++ implementation consists of two executables:

- `atst_sed_worker` is a persistent callback process. It reads one JSON request
  per line from standard input, decodes media through FFmpeg, performs 10-second
  TensorRT inference, and writes JSON callbacks to standard output.
- `atst_sed_testbed` is a Qt 6 frontend. It provides a multi-select media file
  picker and playlist, plays audio/video, and displays the three aggregate
  classes loaded by the worker. A class and percentage turn red above the
  selected threshold.

All C++ sources are in this directory. The TensorRT engine consumes normalized
ATST mel tensors shaped `[1,1,64,1001]` and returns `[1,447,250]` source
probabilities. At startup, the worker reads `class_mapping.csv` and the ordered
`ATST-F_strong_1.labels.txt` vocabulary produced by the converter. It requires
exactly three aggregate classes, sums each class's configured sources, and caps
the result at 100%.

## Convert the model

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
  --labels /workspace/resources/ATST-F_strong_1.labels.txt
```

The Qt process needs access to the AGX display and audio service. When launched
inside Docker, pass through the appropriate X11/Wayland socket and display
environment for the desktop session.

## Worker callback protocol

The worker can also be tested without the frontend. It first emits a `ready`
callback; requests and callbacks use newline-delimited JSON:

```text
request:  {"id":1,"path":"/workspace/videos/example.mp4"}
callback: {"event":"ready","classes":["Glass breaking","Gunshot","Siren"],"mapping":"/workspace/class_mapping.csv"}
callback: {"event":"started","id":1,"path":"..."}
callback: {"event":"chunk","id":1,"start_ms":0,"hop_ms":40,"scores":[[0.01,0.72,0.03],...]}
callback: {"event":"complete","id":1,"duration_ms":10000}
```
