Pending Release Notes
=====================


Updates / New Features
----------------------

Code Updates

* Removed gen_perturber_combinations from GenerateBlackboxResponse class, so it is now a standalone function.

CI/CD

* Major overhaul of pipeline to improve efficiency and `yml` readability.

* Added `ruff` and `black` check to CI/CD (currently optional).

* Updated coverage to look at `src/nrtk` rather than `nrtk`.

Other

* Added `git pre-hook` to assist in linting.

* Refactored package into `src/nrtk` instead of `nrtk`.

* Add `prefer-active-python=true` to `poetry.toml` to use system `Python`.

Dependencies

* Added new linting `black` and `ruff`.

Documentation

* Updated documents to reflect new refactor.

Fixes
-----
