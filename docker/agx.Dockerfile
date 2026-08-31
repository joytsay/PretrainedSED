# Jetson AGX Orin / JetPack 6.2.1 / L4T 36.4.4
# Build this image on the Jetson itself (ARM64).
FROM nvcr.io/nvidia/pytorch:25.06-py3-igpu

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    QT_X11_NO_MITSHM=1 \
    QT_MEDIA_BACKEND=gstreamer \
    GST_PLUGIN_FEATURE_RANK=nvv4l2decoder:0 \
    XDG_RUNTIME_DIR=/tmp/runtime-root \
    PULSE_SERVER=unix:/run/user/1000/pulse/native \
    PULSE_COOKIE=/root/.config/pulse/cookie

RUN touch /etc/modules \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg \
      libsndfile1 \
      libsndfile1-dev \
      libglib2.0-0 \
      libasound2-plugins \
      git \
      build-essential \
      cmake \
      ninja-build \
      kmod \
      nlohmann-json3-dev \
      qt6-base-dev \
      qt6-multimedia-dev \
      libxcb-cursor0 \
      gstreamer1.0-tools \
      gstreamer1.0-libav \
      gstreamer1.0-plugins-base \
      gstreamer1.0-plugins-good \
      gstreamer1.0-plugins-bad \
      gstreamer1.0-plugins-ugly \
      pulseaudio-utils \
      libsox-dev \
      libsox-fmt-all \
      ca-certificates \
 && install -d -m 0700 /tmp/runtime-root \
 && rm -rf /var/lib/apt/lists/*

# The Jetson image provides Torch 2.8, but does not ship a compatible
# torchaudio wheel. Build the matching ARM64/Python-3.12 package.
RUN python3 -m pip install --upgrade pip 'setuptools<81' wheel \
 && git clone --recursive --depth=1 -b release/2.8 \
      https://github.com/pytorch/audio.git /tmp/audio \
 && cd /tmp/audio \
 && sed -i '1i#include <float.h>' \
      src/libtorchaudio/cuctc/src/ctc_prefix_decoder_kernel_v2.cu \
 && BUILD_VERSION=2.8.0 BUILD_SOX=1 TORCH_CUDA_ARCH_LIST=8.7 \
      MAX_JOBS=4 python3 setup.py bdist_wheel --dist-dir /tmp/torchaudio-dist \
 && python3 -m pip install /tmp/torchaudio-dist/torchaudio-2.8.0-cp312-cp312-linux_aarch64.whl --no-deps \
 && rm -rf /tmp/audio /tmp/torchaudio-dist

WORKDIR /workspace

# The base image supplies the Jetson-compatible CUDA/PyTorch stack. Install
# only this repository's Python dependencies; do not replace torch here.
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip 'setuptools<81' wheel \
 && python3 -m pip install "numpy<2" \
 && grep -v '^sed_scores_eval==' /tmp/requirements.txt > /tmp/requirements-jetson.txt \
 && python3 -m pip install -r /tmp/requirements-jetson.txt \
 && python3 -m pip install --no-build-isolation sed_scores_eval==0.0.3 \
 && rm -f /tmp/requirements.txt /tmp/requirements-jetson.txt

CMD ["bash"]
