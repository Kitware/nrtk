* Ensure input and output boxes do not share memory

* Fixed a bug in all noise perturbers (``GaussianNoisePerturber``,
  ``PepperNoisePerturber``, ``SaltAndPepperNoisePerturber``,
  ``SaltNoisePerturber``, ``SpeckleNoisePerturber``) where the return
  value of ``super().perturb()`` was discarded, causing output boxes to
  share identity with the input boxes instead of being deep-copied.
