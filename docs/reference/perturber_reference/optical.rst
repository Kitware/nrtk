##################
Optical Perturbers
##################

Optical perturbers simulate physics-based sensor and optical effects using the pyBSM (Python-based Sensor Model)
library. These perturbations model real sensor behavior, aperture characteristics, and optical phenomena to support
realistic, sensor-driven robustness evaluations. For detailed information about pyBSM concepts and parameters, see the
`pyBSM documentation <https://pybsm.readthedocs.io/en/latest/explanation.html>`_.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Perturbation
     - Required Inputs
     - Optional Parameters

   * - :ref:`CircularApertureOTFPerturber`
     - Image (RGB/Grayscale)
     - * ``sensor``
       * ``scenario``
       * ``mtf_wavelengths``
       * ``mtf_weights``
       * ``interp``
       * ``boxes``
       * ``img_gsd``

   * - :ref:`DefocusOTFPerturber`
     - Image (RGB/Grayscale)
     - * ``sensor``
       * ``scenario``
       * ``w_x``
       * ``w_y``
       * ``interp``
       * ``boxes``
       * ``img_gsd``

   * - :ref:`DetectorOTFPerturber`
     - Image (RGB/Grayscale)
     - * ``sensor``
       * ``scenario``
       * ``w_x``
       * ``w_y``
       * ``f``
       * ``interp``
       * ``boxes``
       * ``img_gsd``

   * - :ref:`JitterOTFPerturber`
     - Image (RGB/Grayscale)
     - * ``sensor``
       * ``scenario``
       * ``s_x``
       * ``s_y``
       * ``interp``
       * ``boxes``
       * ``img_gsd``

   * - :class:`~nrtk.impls.perturb_image.optical.pybsm_perturber.PybsmPerturber`
     - Image (RGB/Grayscale)
     - * ``sensor``
       * ``scenario``
       * ``reflectance_range``
       * ``rng_seed``
       * ``boxes``
       * ``img_gsd``

   * - :ref:`RadialDistortionPerturber`
     - Image (RGB/Grayscale)
     - * ``k``
       * ``color_fill``
       * ``boxes``

   * - :ref:`TurbulenceApertureOTFPerturber`
     - Image (RGB/Grayscale)
     - * ``sensor``
       * ``scenario``
       * ``mtf_wavelengths``
       * ``mtf_weights``
       * ``altitude``
       * ``slant_range``
       * ``D``
       * ``ha_wind_speed``
       * ``cn2_at_1m``
       * ``int_time``
       * ``n_tdi``
       * ``aircraft_speed``
       * ``interp``
       * ``boxes``
       * ``img_gsd``
