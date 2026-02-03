##################
Wrapper Perturbers
##################

Wrapper perturbers enable composition of multiple perturbations or provide integration with third-party augmentation
libraries. These perturbers facilitate complex perturbation pipelines and leverage external tools for additional
transformation capabilities.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Perturbation
     - Required Inputs
     - Optional Parameters

   * - :class:`~nrtk.impls.perturb_image.wrapper.AlbumentationsPerturber`
     - Image (format varies by transform)
     - * ``perturber``
       * ``parameters``
       * ``seed``
       * ``boxes``

   * - :class:`~nrtk.impls.perturb_image.wrapper.ComposePerturber`
     - Image (format varies by perturbers)
     - * ``perturbers``
       * ``boxes``
