* Refactored ``PerturbImageFactory`` tests to use abstract ``_make_factory`` method pattern.

* Moved numpy array conversion from ``PerturberMultivariateFactory.from_config`` to pyBSM perturber
  ``__init__`` methods.

* Introduced ``_TestPerturbImageFactory`` base test class with shared test cases for all factory implementations.

* Replaced dependency on perturber implementations requiring extras with ``FakePerturber`` in factory tests.

* Added ``PerturberFakeFactory`` to ``tests/fakes.py`` for testing factory interface behavior.

* Added ``PerturbImageFactory`` interface tests in ``tests/interfaces/test_perturb_image_factory.py``.

* Fixed ``WaterDropletPerturber`` ``test_regression_get_random_points_within_min_dist`` to use its own RNG
  for deterministic results.
