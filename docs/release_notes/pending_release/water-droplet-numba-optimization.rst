* Replaced ``geopandas`` and ``shapely`` dependencies in ``WaterDropletPerturber`` with a
  Numba-accelerated ray casting algorithm for point-in-polygon tests. This reduces the
  dependency footprint and improves performance for high droplet counts.

* Refactored numba JIT functions in ``WaterDropletPerturber`` to separate pure Python
  implementations (``_points_in_polygon_impl``, ``_compute_refraction_mapping_impl``) from
  JIT wrappers. This improves testability and code coverage while maintaining performance
  when numba is available.

* Added Protocol type hints for the JIT-compiled functions to ensure proper static type
  checking without requiring type ignores at call sites.

* Removed unused ``_to_sphere_section_env`` method that was replaced by the vectorized
  ``_compute_refraction_mapping`` function.

* Disabled caching for notebook jobs to improve length of job time taken.

* Renamed internal functions containing ``__`` to ``_``.
