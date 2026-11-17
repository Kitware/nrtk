* Added ``JitterVideoPerturber`` for physics-based line-of-sight jitter simulation on video using
  SciPy, with cited per-platform presets loaded from bundled JSON files and a per-frame ``jitter``
  record of the achieved degradation.

* Added ``scipy`` as an optional dependency and removed it from the ``hcipy`` extra;
  ``TurbulenceVideoPerturber`` now requires ``nrtk[hcipy,scipy]``.
