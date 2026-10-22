# Audio-only CUDA image. The named source context is the audio.cpp checkout.
# No embedded WebUI build, Node, prompt data, llama.cpp, or RAG workers.
ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=12.9.2
ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A
ARG GCC_VERSION=14

ARG BASE_CUDA_DEV_CONTAINER=docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG BASE_CUDA_RUN_CONTAINER=docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

FROM ${BASE_CUDA_DEV_CONTAINER} AS build

ARG GCC_VERSION=14

# Install build toolchain
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc-${GCC_VERSION} g++-${GCC_VERSION} cmake ca-certificates ccache

ENV CC=gcc-${GCC_VERSION} CXX=g++-${GCC_VERSION} CUDAHOSTCXX=g++-${GCC_VERSION}
ENV CCACHE_DIR=/root/.cache/ccache CCACHE_MAXSIZE=10G CCACHE_COMPILERCHECK=content

ARG AUDIOCPP_VERSION=dev
ARG AUDIOCPP_MODEL_SET=full
ARG AUDIOCPP_MODELS=
ARG AUDIOCPP_CPU_ALL_VARIANTS=ON
ARG BUILD_JOBS=0
# CUDA architectures to compile for.
# - default = the portable default list from CMakeLists.txt
# - for a custom arch set build with --build-arg CUDA_DOCKER_ARCH="89-real;...".
ARG CUDA_DOCKER_ARCH=default

WORKDIR /app
# Keep dependency sources in a stable layer when application/UI files change.
# Copy only native build inputs here; SIP and WebUI sources have their own stages.
COPY --from=audiocpp-source external/ ./external/
COPY --from=audiocpp-source include/ ./include/
COPY --from=audiocpp-source src/ ./src/
COPY --from=audiocpp-source app/ ./app/
COPY --from=audiocpp-source tools/ ./tools/
COPY --from=audiocpp-source model_specs/ ./model_specs/
COPY --from=audiocpp-source assets/ ./assets/
# model_perf is an unconditional release target, despite living under tests/.
COPY --from=audiocpp-source tests/perf/ ./tests/perf/
COPY --from=audiocpp-source CMakeLists.txt ./
COPY --from=audiocpp-source .devops/entrypoint.sh ./.devops/entrypoint.sh
COPY --from=audiocpp-source webui/native/demo_voices/ ./webui/native/demo_voices/

RUN --mount=type=cache,target=/app/build,sharing=locked \
    --mount=type=cache,id=audiocpp-cuda-ccache,target=/root/.cache/ccache \
    if [ "${CUDA_DOCKER_ARCH}" != "default" ]; then \
        ADDITIONAL_CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}"; \
    fi && \
    cmake -S . -B build \
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
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined && \
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

FROM ${BASE_CUDA_RUN_CONTAINER} AS base

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
COPY --from=audiocpp-source model_specs/ /app/model_specs/
COPY --from=audiocpp-source assets/framework/models/ /app/assets/framework/models/
COPY --from=audiocpp-source tools/model_manager_v2.py /app/tools/model_manager_v2.py

COPY --from=build /app/webui/native/demo_voices/demo_1_man.wav /app/preset-voices/man.wav
COPY voices/ /app/preset-voices/

RUN mkdir -p /app/models /app/voices && chown ubuntu:ubuntu /app/models /app/voices

USER ubuntu

ENTRYPOINT ["/app/entrypoint.sh"]

