* Added import guards for noise perturbers (``GaussianNoisePerturber``, ``PepperNoisePerturber``,
  ``SaltAndPepperNoisePerturber``, ``SaltNoisePerturber``, ``SpeckleNoisePerturber``). Attempting to use these classes
  without ``scikit-image`` installed now raises a clear ``ImportError`` with install instructions instead of an opaque
  failure. Restructured into private ``_noise/`` package with public exports from
  ``nrtk.impls.perturb_image.photometric.noise``.
