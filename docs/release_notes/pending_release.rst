Pending Release Notes
=====================

Updates / New Features
----------------------

New Features

* Implemented functionallity for computing the NIIRS metric as a ImageMetrics implementation

* Added name property to ImageMetrics implementation

* Implemented Circular Aperture OTF perturber

* Modified to optionally return thetas as floats or ints

* Implemented Detector OTF perturber

CI/CD

* Added checks and tests for OpenCV perturbers to ensure image is in channel-last format.

* Added a mirroring job to replace builtin gitlab mirroring due to LFS issue.

* Removed an old opencv version check script.

* Added `syrupy` dependency for snapshot-based regression testing.

Documentation

* Updated the readthedocs required packages.

* Removed lfs objects that were in `docs` to allow readthedocs to render.

Fixes
-----
