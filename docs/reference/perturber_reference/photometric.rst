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

   * - :ref:`AverageBlurPerturber`
     - Image (RGB/Grayscale)
     - * ``ksize``
       * ``boxes``

   * - :ref:`BrightnessPerturber`
     - Image (RGB)
     - * ``factor``
       * ``boxes``

   * - :ref:`ColorPerturber`
     - Image (RGB)
     - * ``factor``
       * ``boxes``

   * - :ref:`ContrastPerturber`
     - Image (RGB)
     - * ``factor``
       * ``boxes``

   * - :ref:`GaussianBlurPerturber`
     - Image (RGB/Grayscale)
     - * ``ksize``
       * ``boxes``

   * - :ref:`GaussianNoisePerturber`
     - Image (RGB/Grayscale)
     - * ``rng``
       * ``mean``
       * ``var``
       * ``boxes``

   * - :ref:`MedianBlurPerturber`
     - Image (RGB/Grayscale)
     - * ``ksize``
       * ``boxes``

   * - :ref:`PepperNoisePerturber`
     - Image (RGB/Grayscale)
     - * ``rng``
       * ``amount``
       * ``boxes``

   * - :ref:`SaltAndPepperNoisePerturber`
     - Image (RGB/Grayscale)
     - * ``rng``
       * ``amount``
       * ``salt_vs_pepper``
       * ``boxes``

   * - :ref:`SaltNoisePerturber`
     - Image (RGB/Grayscale)
     - * ``rng``
       * ``amount``
       * ``boxes``

   * - :ref:`SharpnessPerturber`
     - Image (RGB)
     - * ``factor``
       * ``boxes``

   * - :ref:`SpeckleNoisePerturber`
     - Image (RGB/Grayscale)
     - * ``rng``
       * ``mean``
       * ``var``
       * ``boxes``
