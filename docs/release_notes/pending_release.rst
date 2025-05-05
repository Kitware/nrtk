Pending Release Notes
=====================

Updates / New Features
----------------------
* PyBSM based PerturbImage implementations now scale provided bounding boxes

* JATICDetectionAugmentation now uses configured image perturber to update
  provided bounding boxes

* Updated test_blur to use Syrupy for image comparison.

* Updated test_enhance to use Syrupy for image comparison.

CI/CD

* Added pyright static checking for example jupyter notebooks under ``tests/examples``.

* Removed dependency on ``maite`` for static type checking.

* Removed ``mypy`` check and dependency.

* Updated RTD build environment to match Gitlab Pages environment.

Documentation

* Added new ``otf_visualization`` notebooks for existing OTF perturbers

* Updated ``otf_examples.rst`` to render notebooks in docs

* Updated ``README.md``, ``getting_started.rst``, ``index.rst``, and ``installation.rst`` as part of Diataxis refactor.

* Added ``nrtk-explorer`` section to ``README.md``.

* Corrected Google Colab links in example notebooks

* Updated ``index.rst``, ``installation.rst``, and ``README.md``  based on ``devel-jatic``.

* Added ``Extras`` section to ``installation.rst``.

* Replaced ``introduction.rst``  with ``nrtk_explanation.rst`` for Explanation section of docs.

* Replaced JATIC Gitlab links with public-facing links in ``README.md``.

* Added ``nrtk_how_to_topics.rst`` to documentation.

* Added ``glossary.rst`` to documentation.

* Added generic ``AlbumentationsPerturber`` to use perturbers from the Albumentations module

* Added ``ROADMAP.md``.

* Added a ``Containers`` section to documentation

* Added ``AUKUS.rst`` to Containers documentations

* Updated T&E notebook titles.

* Added ``nrtk_lens_flare_demo.ipynb`` to documentation.

* Added warning to use Poetry only in a virtual environment per Poetry documentation.

Fixes
-----

* Fixed incorrect link for ``nrtk_tutorial``.

* Fixed missing extras install for notebooks CI.

* Fixed error with ``translation_perturber`` when ``max_translation_limit`` is ``(0, 0)``.

Examples
--------
* Added ``nrtk_sensor_transformation_demo`` notebook from ``nrtk_jatic``

* Added an example notebook exploring the HazePerturber and its use.

* Added an example notebook demonstrating use of the new ``AlbumentationsPerturber``

* Added a guide exploring a lens flare perturber using Albumentations
