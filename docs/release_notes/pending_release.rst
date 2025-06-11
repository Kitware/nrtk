Pending Release Notes
=====================

Updates / New Features
----------------------

Examples

* Added an example notebook guide to demonstrate the use of the ``WaterDropletPerturber``.

Documentation

* Improved inline documentation and docstring formatting for files under
  ``src/nrtk/perturb_image/impls/generic`` and ``src/nrtk/perturb_image/impls/pybsm``.

* Improved installation documentation for README.md

* Added a new section to the installation guide that lists the key dependencies for each perturber.

* Added a documentation page outlining operational risks

* Ensure intra-documentation links are consistent

* Added explanatory context to figure 1 in ``nrtk_explanation.rst``.

* Added missing T&E notebooks to ``nrtk_jatic/testing_and_eval_guides.rst``.

* Added ``WaterDropletPerturber`` to Risk Factors Table in ``risk_factors.rst``.

* Improved documentation based on Phase-1 documentation feedback.

Fixes
-----

* Fixed pytest-core CI job with import guards for MAITE and notebook releated tests.

* Fixed ``LinspaceStepPerturber`` to follow default linspace behavior (endpoint=True)

* Fixed errors in T&E Guides and added Colab link.

* Fixed some broken URLs in jupyter notebooks

* Generalized ``DummyPerturber`` class to be used in tests.

* Use context handlers for file opens in ``tests/impls/score_detections/test_coco_scorer.py``

* Fixed ruff rules and updated linting.

* Removed ``Union`` and ``Optional`` type hints.

* Improved the completeness score of pyright --verifytypes

* Enabled ``D`` flag for ruff and fixed associated errors.
