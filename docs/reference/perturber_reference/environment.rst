######################
Environment Perturbers
######################

Environment perturbers simulate atmospheric and weather-related effects that occur in real-world imaging conditions,
such as haze, fog, and water droplets on camera lenses. These perturbations model environmental degradation of image
quality.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Perturbation
     - Required Inputs
     - Optional Parameters

   * - :ref:`HazePerturber`
     - Image (RGB/Grayscale)
     - * ``factor``
       * ``depth_map``
       * ``sky_color``
       * ``boxes``

   * - :ref:`WaterDropletPerturber`
     - Image (RGB/Grayscale)
     - * ``size_range``
       * ``num_drops``
       * ``blur_strength``
       * ``psi``
       * ``n_air``
       * ``n_water``
       * ``f_x``
       * ``f_y``
       * ``seed``
       * ``boxes``
