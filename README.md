<!-- :auto badges: -->
[![PyPI - Python Version](https://img.shields.io/pypi/v/nrtk)](https://pypi.org/project/nrtk/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/nrtk)
[![Documentation Status](https://readthedocs.org/projects/nrtk/badge/?version=latest)](https://nrtk.readthedocs.io/en/latest/?badge=latest)
<!-- :auto badges: -->

# Natural Robustness Toolkit (NRTK)

The `nrtk` package is an open source toolkit for evaluating the natural robustness of computer vision
algorithms to various perturbations, including sensor-specific changes to camera focal length, aperture
diameter, etc. Functionality is provided through [Strategy](https://en.wikipedia.org/wiki/Strategy_pattern)
and [Adapter](https://en.wikipedia.org/wiki/Adapter_pattern) patterns to allow for modular integration
into systems and applications.

We have also created the [`nrtk-jatic`](https://github.com/Kitware/nrtk-jatic) package to support AI T&E
use-cases and workflows, through interoperability with the [`maite`](https://github.com/mit-ll-ai-technology/maite)
library and integration with other [JATIC](https://cdao.pages.jatic.net/public/) tools. Users seeking to use NRTK to
perturb MAITE-wrapped datasets or evaluate MAITE-wrapped models should
start with the `nrtk-jatic` package.

<!-- :auto installation: -->
## Installation
Ensure the source tree is acquired locally before proceeding.

To install the current version via `pip`:
```bash
pip install nrtk
```

Alternatively, you can use [Poetry](https://python-poetry.org/):
```bash
poetry install
```

Certain plugins may require additional runtime dependencies. Details on these requirements can be found [here](https://nrtk.readthedocs.io/en/latest/implementations.html).

For more detailed installation instructions, visit the [installation documentation](https://nrtk.readthedocs.io/en/latest/installation.html).
<!-- :auto installation: -->

<!-- :auto getting-started: -->
## Getting Started
Explore usage examples of the `nrtk` package in various contexts using the Jupyter notebooks provided in the `./examples/` directory.

Contributions are encouraged! For more details, refer to the [CONTRIBUTING.md](./CONTRIBUTING.md) file.
<!-- :auto getting-started: -->

<!-- :auto documentation: -->
## Documentation
Documentation for both release snapshots and the latest master branch is available on [ReadTheDocs](https://nrtk.readthedocs.io/en/latest/).

To build the Sphinx-based documentation locally for the latest reference:
```bash
# Install dependencies
poetry install --sync --with linting,tests,docs
# Navigate to the documentation root
cd docs
# Build the documentation
poetry run make html
# Open the generated documentation in your browser
firefox _build/html/index.html
```
<!-- :auto documentation: -->

<!-- :auto developer-tools: -->
## Developer Tools

### Pre-commit Hooks
Pre-commit hooks ensure that code complies with required linting and formatting guidelines. These hooks run automatically before commits but can also be executed manually. To bypass checks during a commit, use the `--no-verify` flag.

To install and use pre-commit hooks:
```bash
# Install required dependencies
poetry install --sync --with linting,tests,docs
# Initialize pre-commit hooks for the repository
poetry run pre-commit install
# Run pre-commit checks on all files
poetry run pre-commit run --all-files
```
<!-- :auto developer-tools: -->

<!-- :auto contributing: -->
## Contributing
- Follow the [JATIC Design Principles](https://cdao.pages.jatic.net/public/program/design-principles/).
- Adopt the Git Flow branching strategy.
- Detailed release information is available in [docs/release_process.rst](./docs/release_process.rst).
- Additional contribution guidelines can be found in [CONTRIBUTING.md](./CONTRIBUTING.md).
<!-- :auto contributing: -->

<!-- :auto license: -->
## License
[Apache 2.0](./LICENSE)
<!-- :auto license: -->

<!-- :auto contacts: -->
## Contacts

**Principal Investigator**: Brian Hu (Kitware) @brian.hu
**Product Owner**: Austin Whitesell (MITRE) @awhitesell
**Scrum Master / Tech Lead**: Brandon RichardWebster (Kitware) @b.richardwebster
**Deputy Tech Lead**: Emily Veenhuis (Kitware) @emily.veenhuis
<!-- :auto contacts: -->