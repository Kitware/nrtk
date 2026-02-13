######################
Photometric Perturbers
######################

Photometric perturbers modify the visual appearance of images by adjusting color, brightness, contrast, sharpness,
blur, and noise properties. These transformations simulate variations in lighting conditions, camera settings, and
image quality degradation.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Perturbation
     - Required Inputs
     - Optional Parameters

   * - :class:`~nrtk.impls.perturb_image.photometric.blur.AverageBlurPerturber`
     - Image (RGB/Grayscale)
     - * ``ksize``
       * ``boxes``

   * - :class:`~nrtk.impls.perturb_image.photometric.enhance.BrightnessPerturber`
     - Image (RGB)
     - * ``factor``
       * ``boxes``

   * - :class:`~nrtk.impls.perturb_image.photometric.enhance.ColorPerturber`
     - Image (RGB)
     - * ``factor``
       * ``boxes``

   * - :class:`~nrtk.impls.perturb_image.photometric.enhance.ContrastPerturber`
     - Image (RGB)
     - * ``factor``
       * ``boxes``

   * - :class:`~nrtk.impls.perturb_image.photometric.blur.GaussianBlurPerturber`
     - Image (RGB/Grayscale)
     - * ``ksize``
       * ``boxes``

   * - :class:`~nrtk.impls.perturb_image.photometric.noise.GaussianNoisePerturber`
     - Image (RGB/Grayscale)
     - * ``rng``
       * ``mean``
       * ``var``
       * ``boxes``

   * - :class:`~nrtk.impls.perturb_image.photometric.blur.MedianBlurPerturber`
     - Image (RGB/Grayscale)
     - * ``ksize``
       * ``boxes``

   * - :class:`~nrtk.impls.perturb_image.photometric.noise.PepperNoisePerturber`
     - Image (RGB/Grayscale)
     - * ``rng``
       * ``amount``
       * ``boxes``

   * - :class:`~nrtk.impls.perturb_image.photometric.noise.SaltAndPepperNoisePerturber`
     - Image (RGB/Grayscale)
     - * ``rng``
       * ``amount``
       * ``salt_vs_pepper``
       * ``boxes``

   * - :class:`~nrtk.impls.perturb_image.photometric.noise.SaltNoisePerturber`
     - Image (RGB/Grayscale)
     - * ``rng``
       * ``amount``
       * ``boxes``

   * - :class:`~nrtk.impls.perturb_image.photometric.enhance.SharpnessPerturber`
     - Image (RGB)
     - * ``factor``
       * ``boxes``

   * - :class:`~nrtk.impls.perturb_image.photometric.noise.SpeckleNoisePerturber`
     - Image (RGB/Grayscale)
     - * ``rng``
       * ``mean``
       * ``var``
       * ``boxes``
