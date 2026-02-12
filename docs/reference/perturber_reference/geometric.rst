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

   * - :class:`~nrtk.impls.perturb_image.geometric.random.RandomCropPerturber`
     - Image (RGB/Grayscale)
     - * ``crop_size``
       * ``seed``
       * ``boxes``

   * - :class:`~nrtk.impls.perturb_image.geometric.random.RandomRotationPerturber`
     - Image (RGB)
     - * ``limit``
       * ``probability``
       * ``fill``
       * ``seed``
       * ``boxes``

   * - :class:`~nrtk.impls.perturb_image.geometric.random.RandomScalePerturber`
     - Image (RGB)
     - * ``limit``
       * ``interpolation``
       * ``probability``
       * ``seed``
       * ``boxes``

   * - :class:`~nrtk.impls.perturb_image.geometric.random.RandomTranslationPerturber`
     - Image (RGB/Grayscale)
     - * ``seed``
       * ``color_fill``
       * ``max_translation_limit``
       * ``boxes``
