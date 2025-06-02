Pending Release Notes
=====================

Updates / New Features
----------------------

* Added a new section to the installation guide that lists the key dependencies for each perturber.

* Added a documentation page outlining operational risks

* Improved installation documentation for README.md

Examples
--------

* Added an example notebook guide to demonstrate the use of the ``WaterDropletPerturber``.

Fixes
-----

* Fixed pytest-core CI job with import guards for MAITE and notebook releated tests.

* Fixed ``LinspaceStepPerturber`` to follow default linspace behavior (endpoint=True)

* Fixed errors in T&E Guides and added Colab link.

* Fixed some broken URLs in jupyter notebooks

* Generalized ``DummyPerturber`` class to be used in tests.

* Use context handlers for file opens in ``tests/impls/score_detections/test_coco_scorer.py``
