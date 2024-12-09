Pending Release Notes
=====================

Updates / New Features
----------------------

CI/CD

* Updated and applied ruff configuration.

Interfaces

* Exposed random seed parameter to ``PybsmPerturber`` allowing users to control the randomness
  of pybsm pertubations

* Updated ``perturb`` to include optional ``boxes`` argument which contains the bounding boxes for the given image.

* Updated ``perturb`` to return the image in addition to the modified bounding boxes.

Fixes
-----
