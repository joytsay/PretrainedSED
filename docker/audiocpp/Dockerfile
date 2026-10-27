# Audio-only CUDA image built from the native source snapshot in source/.
# No embedded WebUI build, Node, prompt data, llama.cpp, or RAG workers.
ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A
ARG GCC_VERSION=11

# Share one Jetson development base between compile and runtime stages.
# Override with a CUDA 12.6 / Ubuntu 22.04 image built by jetson-containers.
ARG BASE_CUDA_CONTAINER=nvcr.io/nvidia/l4t-jetpack:r36.4.0

FROM ${BASE_CUDA_CONTAINER} AS build

ARG GCC_VERSION=11

# Install build toolchain
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc-${GCC_VERSION} g++-${GCC_VERSION} cmake ca-certificates ccache \
        cuda-cudart-dev-12-6

ENV CC=gcc-${GCC_VERSION} CXX=g++-${GCC_VERSION} CUDAHOSTCXX=g++-${GCC_VERSION}
ENV CCACHE_DIR=/root/.cache/ccache CCACHE_MAXSIZE=10G CCACHE_COMPILERCHECK=content

ARG AUDIOCPP_VERSION=dev
ARG AUDIOCPP_MODEL_SET=full
ARG AUDIOCPP_MODELS=
ARG AUDIOCPP_CPU_ALL_VARIANTS=ON
ARG BUILD_JOBS=1
# CUDA architectures to compile for.
# - default = the portable default list from CMakeLists.txt
# - for a custom arch set build with --build-arg CUDA_DOCKER_ARCH="89-real;...".
ARG CUDA_DOCKER_ARCH=87-real

WORKDIR /app
# Keep dependency sources in a stable layer when application/UI files change.
# Copy only native build inputs here; SIP and WebUI sources have their own stages.
COPY source/external/ ./external/
COPY source/include/ ./include/
COPY source/src/ ./src/
COPY source/app/ ./app/
COPY source/tools/ ./tools/
COPY source/model_specs/ ./model_specs/
COPY source/assets/ ./assets/
# model_perf is an unconditional release target, despite living under tests/.
COPY source/tests/perf/ ./tests/perf/
COPY source/CMakeLists.txt ./
COPY source/.devops/entrypoint.sh ./.devops/entrypoint.sh
COPY source/webui/native/demo_voices/ ./webui/native/demo_voices/

# The cache survives base-image changes: redetect the CUDA toolchain, while
# retaining target objects and ccache. Explicit -L also covers CMake 3.22 CUDA
# runtime linkage, which can emit bare -lcudadevrt/-lcudart_static flags.
RUN --mount=type=cache,target=/app/build,sharing=locked \
    --mount=type=cache,id=audiocpp-cuda-ccache,target=/root/.cache/ccache \
    set -eu; \
    psed_cuda_root="$(readlink -f "${CUDAToolkit_ROOT:-/usr/local/cuda}")"; \
    psed_cuda_static="$(find "${psed_cuda_root}" -name libcudart_static.a -print -quit)"; \
    if [ -z "${psed_cuda_static}" ]; then \
        echo "Missing libcudart_static.a: install cuda-cudart-dev-12-6 in the build image" >&2; exit 1; \
    fi; \
    psed_cuda_lib="$(dirname "${psed_cuda_static}")"; \
    if [ ! -f "${psed_cuda_lib}/libcudadevrt.a" ]; then \
        echo "Missing libcudadevrt.a in ${psed_cuda_lib}: check cuda-cudart-dev-12-6" >&2; exit 1; \
    fi; \
    echo "CUDA toolkit: ${psed_cuda_root}; static runtime libraries: ${psed_cuda_lib}"; \
    mkdir -p build; \
    rm -f build/CMakeCache.txt; \
    if [ -d build/CMakeFiles ]; then \
        find build/CMakeFiles -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' -exec rm -rf -- {} +; \
    fi; \
    ADDITIONAL_CMAKE_ARGS=""; \
    if [ "${CUDA_DOCKER_ARCH}" != "default" ]; then \
        ADDITIONAL_CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}"; \
    fi && \
    cmake -S . -B build \
        -DCUDAToolkit_ROOT="${psed_cuda_root}" \
        -DCMAKE_CUDA_COMPILER="${psed_cuda_root}/bin/nvcc" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        -DAUDIOCPP_MODEL_SET="${AUDIOCPP_MODEL_SET}" \
        -DAUDIOCPP_MODELS="${AUDIOCPP_MODELS}" \
        -DENGINE_ENABLE_CPU_ALL_VARIANTS="${AUDIOCPP_CPU_ALL_VARIANTS}" \
        -DENGINE_ENABLE_CUDA=ON \
        -DENGINE_ENABLE_CUDA_GRAPHS=ON \
        -DGGML_CUDA_NCCL=OFF \
        -DENGINE_ENABLE_VULKAN=OFF \
        -DENGINE_ENABLE_OPENMP=ON \
        -DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON \
        -DAUDIOCPP_VERSION="${AUDIOCPP_VERSION}" \
        -DENGINE_BUILD_EXAMPLES=OFF \
        -DENGINE_BUILD_TESTS=OFF \
        -DENGINE_BUILD_WARMBENCH=OFF \
        ${ADDITIONAL_CMAKE_ARGS} \
        -DCMAKE_EXE_LINKER_FLAGS="-L${psed_cuda_lib} -Wl,--allow-shlib-undefined" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L${psed_cuda_lib}" && \
    if [ "${BUILD_JOBS}" = "0" ]; then BUILD_JOBS="$(nproc)"; fi && \
    cmake --build build --parallel "${BUILD_JOBS}" \
        --target audiocpp_cli \
        --target audiocpp_server \
        --target audiocpp_model_manager \
        --target model_perf && \
    mkdir -p /app/lib /app/full && \
    find build -name "*.so*" -exec cp -P {} /app/lib \; && \
    cp build/bin/audiocpp_cli build/bin/audiocpp_server build/bin/audiocpp_model_manager \
       build/bin/model_perf /app/full/ && \
    cp .devops/entrypoint.sh /app/full/entrypoint.sh && \
    chmod +x /app/full/entrypoint.sh

FROM ${BASE_CUDA_CONTAINER} AS base

ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A
ARG IMAGE_URL=N/A
ARG IMAGE_SOURCE=N/A

LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.version=$APP_VERSION \
      org.opencontainers.image.revision=$APP_REVISION \
      org.opencontainers.image.title="audio.cpp" \
      org.opencontainers.image.description="An all-in-one, pure C++ inference engine for audio models, powered by ggml" \
      org.opencontainers.image.url=$IMAGE_URL \
      org.opencontainers.image.source=$IMAGE_SOURCE

# Runtime deps: OpenMP threading, curl (healthcheck), ffmpeg (audio I/O),
# python3 for the native WebUI's spec-backed model installer, and eSpeak-ng
# for the shared phonemizer used by the native TTS frontends.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 curl ffmpeg python3 ca-certificates espeak-ng libespeak-ng1 && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /tmp/* /var/tmp/* && \
    find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete && \
    find /var/cache -type f -delete

# Docker Hub's CUDA images provide this account, while NVIDIA's Jetson-specific
# L4T CUDA runtime does not. Keep the final image non-root with either base.
RUN if ! id -u ubuntu >/dev/null 2>&1; then \
        useradd --create-home --shell /bin/bash ubuntu; \
    fi

COPY --from=build /app/lib/ /app

WORKDIR /app

# ── FULL: All binaries + entrypoint.sh multiplexer ────────────────────────────
FROM base AS full

COPY --from=build /app/full /app
COPY source/model_specs/ /app/model_specs/
COPY source/assets/framework/models/ /app/assets/framework/models/
COPY source/tools/model_manager_v2.py /app/tools/model_manager_v2.py
COPY source/LICENSE /app/LICENSE.audio.cpp

COPY --from=build /app/webui/native/demo_voices/demo_1_man.wav /app/preset-voices/man.wav
COPY voices/ /app/preset-voices/

RUN mkdir -p /app/models /app/voices && chown ubuntu:ubuntu /app/models /app/voices

USER ubuntu

ENTRYPOINT ["/app/entrypoint.sh"]

