* Default seeding changed from deterministic (``seed=1``) to non-deterministic (``seed=None``)
  for all random perturbers.

* Passing ``numpy.random.Generator`` objects is no longer supported. Use integer seeds instead.

* Added ``RandomPerturbImage`` abstract base class for perturbers using random state.
  This provides standardized seed handling and the new ``is_static`` option.

* Added ``is_static`` parameter to all random perturbers. When ``True`` and a seed
  is provided, the RNG state is reset after each ``perturb()`` call, ensuring identical
  results for repeated calls with the same input.
