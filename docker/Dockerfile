FROM pytorch/pytorch:2.12.1-cuda12.6-cudnn9-devel

ARG MINIFORGE_VERSION=25.3.1-0

ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl bzip2 \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/miniforge.sh \
      "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
 && bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" \
 && rm /tmp/miniforge.sh \
 && conda config --system --set auto_activate_base false \
 && conda clean -afy
