Testing & Evaluation Guides with MAITE
=======================================

Many robustness testing workflows benefit from using NRTK alongside other tools such as the
`JATIC <https://cdao.pages.jatic.net/public/>`_ program's
`Modular AI Trustworthy Engineering (MAITE) <https://github.com/mit-ll-ai-technology/maite>`_ toolbox. While NRTK
focuses on realistic image perturbations, MAITE provides a standardized interface for evaluating model performance
across a set of test conditions. Using these tools together enables modular, reproducible assessments of AI robustness
under simulated operational risks.

The following notebooks showcase how NRTK perturbations can be applied to simulate key
:doc:`operational risks </explanations/risk_factors>` within a
testing and evaluation (T&E) workflow. Each notebook illustrates potential impact on model performance, utilizing MAITE
as an evaluation harness.

----


Photometric Risks
-----------------

Perturbations that alter pixel intensity, color balance, or lighting conditions.

.. grid:: 1 2 2 2
   :gutter: 3
   :padding: 2 2 0 0

   .. grid-item-card:: :material-regular:`light_mode` Extreme Illumination
      :class-card: sd-border-1

      Simulate brightness changes and evaluate model responses under lighting variability.

      +++

      .. button-ref:: /examples/maite/extreme_illumination
         :color: primary
         :outline:

         Open Notebook →

   .. grid-item-card:: :material-regular:`auto_awesome` Lens Flare
      :class-card: sd-border-1

      Simulate a lens flare effect on an image and analyze its average and worst case effects on model precision.

      +++

      .. button-ref:: /examples/maite/lens_flare
         :color: primary
         :outline:

         Open Notebook →

.. seealso::
  Learn how to apply these perturbations in the
  :ref:`Brightness <howto-photometric-brightness>` and
  :ref:`Sun Flare <howto-albumentations-sun-flare>` how-to examples.
  See the :class:`~nrtk.impls.perturb_image.photometric.enhance.BrightnessPerturber` and
  :class:`~nrtk.impls.perturb_image.AlbumentationsPerturber` APIs for parameters.
  For operational context, see
  :doc:`Extreme Illumination </explanations/operational_risk_modules/extreme_illumination>`
  and the related :ref:`Glint / Glare <glint-glare>` risk.

----


Geometric Risks
---------------

Perturbations that change spatial layout through rotation, scaling, or translation.

.. grid:: 1 2 2 2
   :gutter: 3
   :padding: 2 2 0 0

   .. grid-item-card:: :material-regular:`crop_rotate` Affine Transformations
      :class-card: sd-border-1

      Explore how affine transformations affect model inputs and predictions.

      +++

      .. button-ref:: /examples/maite/affine_transformations
         :color: primary
         :outline:

         Open Notebook →

.. seealso::
   For related operational context, see :ref:`Look Angle Different from Training Data <look-angle-variation>`.
   The affine transformations in this tutorial modify images in ways that can be used to evaluate model robustness to
   geometric variations. Consult the
   :ref:`Geometric Perturber APIs <api-geometric-perturbers>` for parameters.

----


Environment Risks
-----------------

Perturbations that replicate weather and atmospheric visibility conditions.

.. grid:: 1 2 2 2
   :gutter: 3
   :padding: 2 2 0 0

   .. grid-item-card:: :material-regular:`cloud` Fog / Haze
      :class-card: sd-border-1

      Evaluate model robustness under haze-like visibility conditions using synthetic perturbations.

      +++

      .. button-ref:: /examples/maite/fog_haze
         :color: primary
         :outline:

         Open Notebook →

   .. grid-item-card:: :material-regular:`water_drop` Rain / Water Droplets
      :class-card: sd-border-1

      Simulate a rain/water droplet effect and analyze its impact on model inputs and predictions.

      +++

      .. button-ref:: /examples/maite/water_droplets
         :color: primary
         :outline:

         Open Notebook →

.. seealso::
   See :doc:`Haze </explanations/operational_risk_modules/haze>` and
   :doc:`Water Droplets on Lens </explanations/operational_risk_modules/water_droplets>`
   for operational context and modeling limitations. Parameter details are in the
   :class:`~nrtk.impls.perturb_image.environment.HazePerturber` and
   :class:`~nrtk.impls.perturb_image.environment.WaterDropletPerturber` APIs.

----


Optical Risks
-------------

Perturbations from sensor optics, camera physics, and atmospheric distortion.

.. grid:: 1 2 2 2
   :gutter: 3
   :padding: 2 2 0 0

   .. grid-item-card:: :material-regular:`center_focus_weak` Visual Focus
      :class-card: sd-border-1

      Apply blur and focus distortions to test performance degradation from defocus.

      +++

      .. button-ref:: /examples/maite/visual_focus
         :color: primary
         :outline:

         Open Notebook →

   .. grid-item-card:: :material-regular:`photo_camera` Resolution & Noise
      :class-card: sd-border-1

      Explore how camera-specific transformations affect model inputs and predictions.

      +++

      .. button-ref:: /examples/maite/sensor_resolution_and_noise
         :color: primary
         :outline:

         Open Notebook →

   .. grid-item-card:: :material-regular:`vibration` Motion Jitter
      :class-card: sd-border-1

      Simulate camera motion jitter and assess its impact on image quality and model inference.

      +++

      .. button-ref:: /examples/maite/motion_jitter
         :color: primary
         :outline:

         Open Notebook →

   .. grid-item-card:: :material-regular:`waves` Atmospheric Turbulence
      :class-card: sd-border-1

      Simulate atmospheric distortion effects and assess their impact on image quality and model inference.

      +++

      .. button-ref:: /examples/maite/atmospheric_turbulence
         :color: primary
         :outline:

         Open Notebook →

   .. grid-item-card:: :material-regular:`lens_blur` Radial Distortion
      :class-card: sd-border-1

      Simulate a radial distortion effect and analyze its impact on model inputs and predictions.

      +++

      .. button-ref:: /examples/maite/radial_distortion
         :color: primary
         :outline:

         Open Notebook →

.. seealso::
  Learn how to apply OTF perturbations in the
  :doc:`Optical Perturbers </examples/optical_perturbers>` how-to guide.
  Use the :doc:`Operational Risk Guide </explanations/risk_factors>` to connect the optical effects in these tutorials
  to deployment conditions, and consult the
  :ref:`Optical Perturber APIs <api-optical-perturbers>`
  for parameter details.

----

.. toctree::
   :hidden:

   /examples/maite/extreme_illumination
   /examples/maite/lens_flare
   /examples/maite/affine_transformations
   /examples/maite/fog_haze
   /examples/maite/water_droplets
   /examples/maite/visual_focus
   /examples/maite/sensor_resolution_and_noise
   /examples/maite/motion_jitter
   /examples/maite/atmospheric_turbulence
   /examples/maite/radial_distortion
