* Added import guards for perturbers with optional dependencies (opencv, albumentations).
  Attempting to use these classes without the required extras now raises clear ``ImportError``
  messages indicating which package to install.

* Blur perturbers (``AverageBlurPerturber``, ``GaussianBlurPerturber``, ``MedianBlurPerturber``)
  now require the ``headless`` or ``graphics`` extra (``opencv-python-headless`` or ``opencv-python``).
  Install with ``pip install nrtk[headless]`` or ``pip install nrtk[graphics]``.

* Geometric perturbers ``RandomRotationPerturber`` and ``RandomScalePerturber`` require the
  ``albumentations`` extra. Install with ``pip install nrtk[albumentations,headless]`` or
  ``pip install nrtk[albumentations,graphics]``.

* Restructured blur perturbers into private ``_blur/`` package with public exports from
  ``nrtk.impls.perturb_image.photometric.blur``.

* Restructured geometric perturbers: ``RandomCropPerturber`` and ``RandomTranslationPerturber``
  moved to private ``_random/`` package with public exports from
  ``nrtk.impls.perturb_image.geometric.random``.

* Restructured albumentations wrapper perturbers into private ``_albumentations/`` package
  with public exports from ``nrtk.impls.perturb_image.wrapper``.

* Added ``tox.ini`` configuration for running tests with specific optional dependency combinations.

* Fixed notebooks to use correct kernel names and data paths for CI compatibility.

* Added ``pytest.importorskip`` guards to test files for graceful skipping when optional
  test dependencies (``httpx``, ``click``) are not installed.
