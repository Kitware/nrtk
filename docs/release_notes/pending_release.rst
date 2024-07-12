Pending Release Notes
=====================

Updates / New Features
----------------------

CI/CD

* Optimized to not run anything but `publish` when `tag`.

* Created a shared `python-version` job for `python` version matrices.

* Updated scanning to properly report the vulnerabilities.

* Updated scanning to properly scan used packages

* Added `from_config` to `PerturbImageFactory` to solve issues hydrating a factory from a json file.


* Added caching of packages to pipeline.

* Changed check release notes to only fetch last commit from main.

* Added examples to `black` scan.

* Added `jupyter` notebook extra to `black`.

* Modified all code to be compliant with all `ruff` and `black` checks besides missing docstrings.

Other

* Added `git pre-hook` to assist in linting.

* Refactored package into `src/nrtk` instead of `nrtk`.

* Add `prefer-active-python=true` to `poetry.toml` to use system `Python`.

* Updated git lfs to properly track large files in any directory.

* Added LinSpacePerturbImageFactory for alternative method of generating pertubations

Dependencies

* Added new linting `black` and `ruff`.

Documentation

* Updated documents to reflect new refactor.

* Added Jitter OTF perturber code doc.

* Added a section that shows visual examples of perturbations based on pyBSM OTF parameters, starting with the Jitter OTF perturber, along with corresponding code snippets to generate these perturbations.

* Added a section to the README about using the pre-commit hooks

Fixes
-----
