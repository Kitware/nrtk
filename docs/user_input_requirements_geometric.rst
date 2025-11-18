####################
Geometric Perturbers
####################

Geometric perturbers alter the spatial positioning and orientation of images through transformations such as rotation,
scaling, cropping, and translation. These perturbations simulate viewpoint changes, camera motion, and framing
variations.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Perturbation
     - Required Inputs
     - Optional Parameters

   * - :ref:`RandomCropPerturber`
     - Image (RGB/Grayscale)
     - * ``crop_size``
       * ``seed``
       * ``boxes``

   * - :ref:`RandomRotationPerturber`
     - Image (RGB)
     - * ``limit``
       * ``probability``
       * ``fill``
       * ``seed``
       * ``boxes``

   * - :ref:`RandomScalePerturber`
     - Image (RGB)
     - * ``limit``
       * ``interpolation``
       * ``probability``
       * ``seed``
       * ``boxes``

   * - :ref:`RandomTranslationPerturber`
     - Image (RGB/Grayscale)
     - * ``seed``
       * ``color_fill``
       * ``max_translation_limit``
       * ``boxes``
