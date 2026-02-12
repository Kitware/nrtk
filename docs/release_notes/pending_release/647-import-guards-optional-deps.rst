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

* Restructured albumentations perturbers into private ``_albumentations/`` package
  with public exports from ``nrtk.impls.perturb_image``.

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

* Added import guards for pyBSM optical perturbers (``PybsmPerturber``,
  ``CircularAperturePerturber``, ``DefocusPerturber``,
  ``DetectorPerturber``, ``JitterPerturber``,
  ``TurbulenceAperturePerturber``).
  Attempting to use these classes without ``pybsm`` installed
  now raises a clear ``ImportError`` with install instructions instead of an
  opaque failure. Restructured into private ``_pybsm/`` package with
  public exports from ``nrtk.impls.perturb_image.optical.otf``.

* Added import guards for MAITE interop module. Attempting to use
  ``MAITEImageClassificationAugmentation`` or
  ``MAITEObjectDetectionAugmentation`` without the ``maite`` extra
  now raises a clear ``ImportError`` with install instructions.
  Restructured with public exports from ``nrtk.interop``.

* Added import guards for ``nrtk.entrypoints``. ``nrtk_perturber``
  requires the ``maite`` extra and ``nrtk_perturber_cli`` requires
  both ``maite`` and ``tools`` extras. Clear ``ImportError`` messages
  are raised when the extras are not installed.

* Added import guards for the MAITE REST API (``nrtk.interop._maite.api``).
  ``handle_post`` and ``handle_aukus_post`` require ``maite`` and ``tools``
  extras.

* **Breaking:** Renamed the ``scikit-image`` extra to ``skimage``. Update install
  commands from ``pip install nrtk[scikit-image]`` to ``pip install nrtk[skimage]``.

* **Breaking:** Renamed the ``Pillow`` extra to ``pillow`` (lowercase). Update install
  commands from ``pip install nrtk[Pillow]`` to ``pip install nrtk[pillow]``.
