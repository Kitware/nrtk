FROM python:3.11-slim as python-base

ENV PYTHONUNBUFFERED=1 \
    # prevents python from creating .pyc
    PYTHONDONTWRITEBUTECODE=1 \
    # pip
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    # poetry
    POETRY_VERSION=1.8.2 \
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

FROM python-base as builder-base
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y curl build-essential libgl1 libglib2.0-0

# install poetry - respects $POETRY_VERSION & $POETRY_HOME \
RUN curl -sSL https://install.python-poetry.org | python3 -

# `development` image is used during development / testing
FROM builder-base as development
ENV FASTAPI_ENV=development
WORKDIR $PYSETUP_PATH

# copy in our built poetry
COPY . ./src
WORKDIR $PYSETUP_PATH/src

# quicker install as runtime deps are already installed
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=cache,target=/root/.cache/pypoetry,sharing=locked \
    poetry config virtualenvs.create false && poetry run pip \
    install -e .[maite-cpu,tools,headless,pybsm-headless]

ENTRYPOINT [ "python", "./src/nrtk_jatic/utils/bin/nrtk_perturber_cli.py"]
# default args for nrtk_perturber_cli
CMD ["/root/input/dataset", "/root/output/", \
     "/root/input/nrtk_config.json", "-v"]

# To run this docker container, use the following command:
# `docker run -v /path/to/input:/root/input/:ro -v /path/to/output:/root/output/ nrtk-perturber`
# This will mount the inputs to the correct locations the default args are used.
# See https://docs.docker.com/storage/volumes/#start-a-container-with-a-volume 
# for more info on mounting volumes
