* Refactored ``**kwargs`` handling from ``PybsmPerturber`` to the shared parent class ``PybsmPerturberMixin``.

* All PyBSM-based perturbers (``DetectorPerturber``, ``TurbulenceAperturePerturber``,
  ``CircularAperturePerturber``, ``JitterPerturber``, ``DefocusPerturber``) now
  accept ``**kwargs`` to modify sensor and scenario parameters.

* Added ``params`` property to ``PybsmPerturberMixin`` to retrieve kwargs passed during initialization.
