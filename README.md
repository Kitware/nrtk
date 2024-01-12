# Natural Robustness Toolkit (NRTK)

## Description
The `nrtk` package is an open source toolkit for evaluating the natural robustness of computer vision
algorithms to various perturbations, including sensor-specific changes to camera focal length, aperture
diameter, etc. Functionality is provided through [Strategy](https://en.wikipedia.org/wiki/Strategy_pattern) and [Adapter](https://en.wikipedia.org/wiki/Adapter_pattern) patterns to allow for modular integration
into systems and applications.

## Installation
The following steps assume the source tree has been acquired locally.

Install the current version via pip:
```bash
pip install nrtk
```

Alternatively, you can also use [Poetry](https://python-poetry.org/):
```bash
poetry install
```

See [here for more installation documentation](
https://nrtk.readthedocs.io/en/latest/installation.html).

## Getting Started
We provide a number of examples based on Jupyter notebooks in the `./examples/` directory to show usage
of the `nrtk` package in a number of different contexts.

## Documentation
Documentation snapshots for releases as well as the latest master are hosted on
[ReadTheDocs](https://nrtk.readthedocs.io/en/latest/).

The sphinx-based documentation may also be built locally for the most
up-to-date reference:
```bash
# Install dependencies
poetry install
# Navigate to the documentation root.
cd docs
# Build the docs.
poetry run make html
# Open in your favorite browser!
firefox _build/html/index.html
```

## Contributing
- We follow the general guidelines outlined in the
[JATIC Software Development Plan](https://gitlab.jatic.net/jatic/docs/sdp/-/blob/main/Branch,%20Merge,%20Release%20Strategy.md).
- We use the Git Flow branching strategy.
- See [docs/release_process.rst](./docs/release_process.rst) for detailed release information.
- See [CONTRIBUTING.md](./CONTRIBUTING.md) for additional contributing information.

## License
Apache 2.0

**POC**: Brian Hu @brian.hu
**DPOC**: Brandon RichardWebster @b.richardwebster
