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
The following steps assume the source tree has been acquired locally.

Install the current version via pip:
```bash
pip install nrtk
```

Alternatively, you can also use [Poetry](https://python-poetry.org/):
```bash
poetry install --sync --with dev-linting,dev-testing,dev-docs
```

See [here for more installation documentation](
https://nrtk.readthedocs.io/en/latest/installation.html).
<!-- :auto installation: -->

<!-- :auto getting-started: -->
## Getting Started
We provide a number of examples based on Jupyter notebooks in the
`./examples/` directory to show usage of the `nrtk` package in a number
of different contexts.
<!-- :auto getting-started: -->

<!-- :auto documentation: -->
## Documentation
Documentation snapshots for releases as well as the latest master are hosted
on [ReadTheDocs](https://nrtk.readthedocs.io/en/latest/).

The sphinx-based documentation may also be built locally for the most
up-to-date reference:
```bash
# Install dependencies
poetry install --sync --with dev-linting,dev-testing,dev-docs
# Navigate to the documentation root.
cd docs
# Build the docs.
poetry run make html
# Open in your favorite browser!
firefox _build/html/index.html
```
<!-- :auto documentation: -->

<!-- :auto developer-tools: -->
## Developer tools

**pre-commit hooks**
pre-commit hooks are used to ensure that any code meets all linting and
formatting guidelines required. After installing, this will always run before
 committing to ensure that any commits are following the standards, but you
 can also manually run the check without committing. If you want to commit
 despite there being errors, you can add `--no-verify` to your commit command.

Installing pre-commit hooks:
```bash
# Ensure that all dependencies are installed
poetry install --sync --with dev-linting,dev-testing,dev-docs
# Initialize pre-commit for the repository
poetry run pre-commit install
# Run pre-commit check on all files
poetry run pre-commit run --all-files
```
<!-- :auto developer-tools: -->

<!-- :auto contributing: -->
## Contributing
- We follow the general guidelines outlined in the
[JATIC Software Development Plan](https://gitlab.jatic.net/jatic/docs/sdp/-/blob/main/Branch,%20Merge,%20Release%20Strategy.md).
- We use the Git Flow branching strategy.
- See [docs/release_process.rst](./docs/release_process.rst) for detailed release information.
- See [CONTRIBUTING.md](./CONTRIBUTING.md) for additional contributing information.
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
