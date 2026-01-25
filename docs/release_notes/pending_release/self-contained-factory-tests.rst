* Refactored ``PerturbImageFactory`` tests to be self-contained. Tests no longer
  depend on ``nrtk.impls`` perturber implementations and instead use
  ``FakePerturber`` from ``tests/fakes.py``.

* Added ``PerturberFactoryMixin`` base class in
  ``tests/impls/perturb_image_factory/`` to provide shared plugin
  discovery tests.
