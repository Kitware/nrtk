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

* Added import guards for enhancement perturbers (``BrightnessPerturber``, ``ColorPerturber``,
  ``ContrastPerturber``, ``SharpnessPerturber``). Attempting to use these classes without
  ``Pillow`` installed now raises a clear ``ImportError`` with install instructions instead of
  an opaque failure. Restructured into private ``_enhance/`` package with public exports from
  ``nrtk.impls.perturb_image.photometric.enhance``.

* Added import guards for diffusion perturbers (``DiffusionPerturber``).
  Attempting to use this class without ``torch``, ``diffusers``, ``Pillow``, or
  ``transformers`` installed now raises a clear ``ImportError`` with install
  instructions instead of an opaque failure. Restructured into private
  ``_diffusion_perturber`` module with public exports from
  ``nrtk.impls.perturb_image.generative``.

* Added ``tox.ini`` configuration for running tests with specific optional dependency combinations.

* Split the ``tox:pytest`` CI job into separate per-Python-version jobs
  (``tox:pytest:py3.10`` through ``tox:pytest:py3.13``) for clearer pipeline organization.

* Fixed notebooks to use correct kernel names and data paths for CI compatibility.

* Added ``pytest.importorskip`` guards to test files for graceful skipping when optional
  test dependencies (``httpx``, ``click``) are not installed.

* Added import guards for water droplet perturber (``WaterDropletPerturber``).
  Attempting to use this class without ``scipy`` and ``numba`` installed
  now raises a clear ``ImportError`` with install instructions instead of an
  opaque failure. Restructured environment perturbers into private modules
  (``_haze_perturber``, ``_water_droplet_perturber``) with public exports from
  ``nrtk.impls.perturb_image.environment``.

* Trimmed redundant ``WaterDropletPerturber`` regression test parametrizations
  from three seeds to one and downscaled the test image from 512x512 to 128x128,
  reducing waterdroplet test suite runtime.

* Renamed private-prefixed functions and protocols in the private
  ``_water_droplet_perturber`` module (``_points_in_polygon_impl``,
  ``_compute_refraction_mapping_impl``, ``_PointsInPolygonProtocol``,
  ``_ComputeRefractionMappingProtocol``) to drop the leading underscores,
  since the module itself is already private.
