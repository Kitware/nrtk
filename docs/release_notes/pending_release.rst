Pending Release Notes
=====================

Updates / New Features
----------------------

CI/CD

* Renamed `linting` job to `flake8`.

* Renamed `typing` job to `mypy`.

* Numerous fixes to the pipeline brought in from `xaitk-jatic`.

* Added a couple TODO notes to the gitlab ci scripts indicating future changes.

* Replaced a `with` to an `only` to force further atomic job environments.

* Swapped out pipeline to use a shared pipeline.

New Features

* Added a compute_image_metrics interface

* Implemented functionality for computing the Signal to Noise Ratio - compute_SNR

* Added bound types for `from_config` definitions 

Fixes
-----
