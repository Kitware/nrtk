* Replaced ``DummyPerturber`` with ``FakePerturber`` in test utilities. The new
  ``FakePerturber`` class in ``tests/fakes.py`` accepts arbitrary keyword arguments,
  making it more flexible for factory tests with various ``theta_key`` values.
