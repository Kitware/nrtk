* Refactored ``**kwargs`` handling from ``PybsmPerturber`` to the shared parent class ``PybsmOTFPerturber``.

* All PyBSM-based perturbers (``DetectorOTFPerturber``, ``TurbulenceApertureOTFPerturber``,
  ``CircularApertureOTFPerturber``, ``JitterOTFPerturber``, ``DefocusOTFPerturber``) now
  accept ``**kwargs`` to modify sensor and scenario parameters.

* Added ``params`` property to ``PybsmOTFPerturber`` to retrieve kwargs passed during initialization.
