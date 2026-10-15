FROM python@sha256:ccc7089399c8bb65dd1fb3ed6d55efa538a3f5e7fca3f5988ac3b5b87e593bf0 AS python-base

ENV PYTHONUNBUFFERED=1 \
    # prevents python from creating .pyc
    PYTHONDONTWRITEBYTECODE=1 \
    # pip
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    # working directory for the application
    PYSETUP_PATH="/opt/pysetup"

# 'builder-base' stage is used to build deps + create virtual environment

FROM python-base AS builder-base
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get -y upgrade \
    && apt-get install -y curl build-essential libgl1 libglib2.0-0 \
    && apt-get purge -y linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip==26.2.1

# 'wheel-builder' stage builds a wheel from local source (used by build-from-source)
FROM builder-base AS wheel-builder
WORKDIR /build
COPY src/ src/
COPY pyproject.toml README.md LICENSE ./
RUN pip install build && python -m build --wheel --outdir /dist

# 'build-from-source' installs nrtk from a locally built wheel
FROM builder-base AS build-from-source
ENV FASTAPI_ENV=development
WORKDIR $PYSETUP_PATH

# Setup numba cache for non-root user
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
RUN mkdir -p /tmp/numba_cache && chmod 777 /tmp/numba_cache

# Install CPU-only PyTorch before nrtk to avoid pulling CUDA (~6 GB savings)
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY --from=wheel-builder /dist /tmp/dist
# Install the locally built wheel by explicit path (not `nrtk[...]`), so pip
# cannot substitute a same-versioned release from PyPI. Extras' transitive
# dependencies still resolve from the index normally.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    WHEEL="$(ls -1 /tmp/dist/nrtk-*.whl | head -n1)" && \
    pip install "${WHEEL}[pybsm,maite,tools,headless,pillow,skimage,albumentations,waterdroplet,diffusion,hcipy,pyav]" && \
    rm -rf /tmp/dist

# Set non-root user
RUN useradd --uid 100 --create-home --shell /bin/bash appuser
USER appuser

# Setup environment variables with default args
ENV INPUT_DATASET_PATH="/input/data/dataset/"
ENV INPUT_DATASET_FORMAT="COCO"
ENV OUTPUT_DATASET_PATH="/output/data/result/"
ENV OUTPUT_DATASET_FORMAT="COCO"
ENV CONFIG_FILE="/input/nrtk_config.json"
ENV COMBINE_OUTPUT="false"
ENV ENABLE_EXPERIMENTAL="false"
ENV OVERWRITE="false"

ENTRYPOINT [ "/usr/local/bin/nrtk-perturber" ]

# 'build-from-pypi' installs nrtk from PyPI (used for release images)
FROM builder-base AS build-from-pypi
ENV FASTAPI_ENV=development
WORKDIR $PYSETUP_PATH

# Setup numba cache for non-root user
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
RUN mkdir -p /tmp/numba_cache && chmod 777 /tmp/numba_cache

# Install CPU-only PyTorch before nrtk to avoid pulling CUDA (~6 GB savings)
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

ARG NRTK_VERSION
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install "nrtk[pybsm,maite,tools,headless,pillow,skimage,albumentations,waterdroplet,diffusion,hcipy,pyav]==${NRTK_VERSION}"

# Set non-root user
RUN useradd --uid 100 --create-home --shell /bin/bash appuser
USER appuser

# Setup environment variables with default args
ENV INPUT_DATASET_PATH="/input/data/dataset/"
ENV INPUT_DATASET_FORMAT="COCO"
ENV OUTPUT_DATASET_PATH="/output/data/result/"
ENV OUTPUT_DATASET_FORMAT="COCO"
ENV CONFIG_FILE="/input/nrtk_config.json"
ENV COMBINE_OUTPUT="false"
ENV ENABLE_EXPERIMENTAL="false"
ENV OVERWRITE="false"

ENTRYPOINT [ "/usr/local/bin/nrtk-perturber" ]

# To run this docker container, use the following command:
# `docker run -v /path/to/input:/input/:ro -v /path/to/output:/output/ nrtk-perturber`.
# Make sure the output directory is writable by non-root users.
# This will mount the inputs to the correct locations the default args are used.
# See https://docs.docker.com/storage/volumes/#start-a-container-with-a-volume
# for more info on mounting volumes
