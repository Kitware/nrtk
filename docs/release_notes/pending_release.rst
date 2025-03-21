Pending Release Notes
=====================

Updates / New Features
----------------------
* PyBSM based PerturbImage implementations now scale provided bounding boxes

* JATICDetectionAugmentation now uses configured image perturber to update
  provided bounding boxes

CI/CD

* Added pyright static checking for example jupyter notebooks under ``tests/examples``.

* Removed dependency on ``maite`` for static type checking.

Documentation

* Added new ``otf_visualization`` notebooks for existing OTF perturbers

* Updated ``otf_examples.rst`` to render notebooks in docs

* Updated ``README.md``, ``getting_started.rst``, ``index.rst``, and ``installation.rst`` as part of Diataxis refactor.

* Added ``nrtk-explorer`` section to ``README.md``.

* Corrected Google Colab links in example notebooks

Fixes
-----

Examples
--------
* Added ``nrtk_sensor_transformation_demo`` notebook from ``nrtk_jatic``
