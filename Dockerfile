FROM python@sha256:5be45dbade29bebd6886af6b438fd7e0b4eb7b611f39ba62b430263f82de36d2 AS python-base

ENV PYTHONUNBUFFERED=1 \
    # prevents python from creating .pyc
    PYTHONDONTWRITEBUTECODE=1 \
    # pip
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    # poetry
    POETRY_VERSION=2.2.1 \
    # change default poetry install location to make it easier to copy to different stages
    POETRY_HOME="/opt/poetry" \
    # install at system level, since in a docker container
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    # do not ask for any interaction
    POETRY_NO_INTERACTION=1 \
    # this is where we copy the lock
    PYSETUP_PATH="/opt/pysetup"

# prepend poetry and venv to path
ENV PATH="$POETRY_HOME/bin:$PATH"

# 'builder-base' stage is used to build deps + create virtual environment

FROM python-base AS builder-base
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y curl build-essential libgl1 libglib2.0-0

# install poetry - respects $POETRY_VERSION & $POETRY_HOME \
RUN curl -sSL https://install.python-poetry.org | python3 -

# `development` image is used during development / testing
FROM builder-base AS development
ENV FASTAPI_ENV=development
WORKDIR $PYSETUP_PATH

# Setup numba cache for non-root user
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
RUN mkdir -p /tmp/numba_cache && chmod 777 /tmp/numba_cache

# copy in our built poetry
COPY . ./src
WORKDIR $PYSETUP_PATH/src

# quicker install as runtime deps are already installed
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=cache,target=/root/.cache/pypoetry,sharing=locked \
    poetry config virtualenvs.create false && poetry run pip \
    install .[pybsm,maite,tools,headless,Pillow,scikit-image,albumentations,waterdroplet,diffusion]

# Set non-root user
RUN useradd --uid 100 --create-home --shell /bin/bash appuser
USER appuser

# Setup environment variables with default args
ENV INPUT_DATASET_PATH="/input/data/dataset/"
ENV OUTPUT_DATASET_PATH="/output/data/result/"
ENV CONFIG_FILE="/input/nrtk_config.json"


ENTRYPOINT [ "/usr/local/bin/nrtk-perturber" ]

# To run this docker container, use the following command:
# `docker run -v /path/to/input:/input/:ro -v /path/to/output:/output/ nrtk-perturber`.
# Make sure the output directory is writable by non-root users.
# This will mount the inputs to the correct locations the default args are used.
# See https://docs.docker.com/storage/volumes/#start-a-container-with-a-volume
# for more info on mounting volumes
