########################
User Input Requirements
########################

This table summarizes the required and optional inputs for each NRTK perturbation. For dependency and installation
requirements, see the :ref:`perturber-dependencies` table.

==================
Generic Perturbers
==================

Generic perturbers apply common image transformations such as blur, brightness, and cropping, and are included in
all NRTK installations for quick, lightweight robustness testing.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Perturbation
     - Required Inputs
     - Optional Parameters

   * - :ref:`AlbumentationsPerturber`
     - Image (format varies by transform)
     - * ``perturber``
       * ``parameters``
       * ``seed``
       * ``boxes``

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

   * - :ref:`ComposePerturber`
     - Image (format varies by perturbers)
     - * ``perturbers``
       * ``boxes``

   * - :ref:`ContrastPerturber`
     - Image (RGB)
     - * ``factor``
       * ``boxes``

   * - :ref:`DiffusionPerturber`
     - Image (converts to RGB)
     - * ``model_name``
       * ``prompt``
       * ``seed``
       * ``num_inference_steps``
       * ``text_guidance_scale``
       * ``image_guidance_scale``
       * ``device``
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

   * - :ref:`HazePerturber`
     - Image (RGB/Grayscale)
     - * ``factor``
       * ``depth_map``
       * ``sky_color``
       * ``boxes``

   * - :ref:`MedianBlurPerturber`
     - Image (RGB/Grayscale)
     - * ``ksize``
       * ``boxes``

   * - :ref:`NOPPerturber`
     - Image (RGB/Grayscale)
     - * ``boxes``

   * - :ref:`PepperNoisePerturber`
     - Image (RGB/Grayscale)
     - * ``rng``
       * ``amount``
       * ``boxes``

   * - :ref:`RadialDistortionPerturber`
     - Image (RGB/Grayscale)
     - * ``k``
       * ``color_fill``
       * ``boxes``

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

================
PyBSM Perturbers
================

PyBSM perturbers simulate physics-based optical effects, such as defocus and diffraction, using the ``pybsm``
extra. These perturbations model real sensor and aperture behavior to support more realistic, sensor-driven
robustness evaluations. For detailed information about pyBSM concepts and parameters, see the
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

   * - :ref:`PybsmPerturber`
     - Image (RGB/Grayscale)
     - * ``sensor``
       * ``scenario``
       * ``reflectance_range``
       * ``rng_seed``
       * ``boxes``
       * ``img_gsd``

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
