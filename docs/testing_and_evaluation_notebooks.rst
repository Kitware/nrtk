Testing & Evaluation Guides with MAITE
--------------------------------------

Many robustness testing workflows benefit from using NRTK alongside other tools such as the
`JATIC <https://cdao.pages.jatic.net/public/>`_ program's
`Modular AI Trustworthy Engineering (MAITE) <https://github.com/mit-ll-ai-technology/maite>`_ toolbox. While NRTK
focuses on realistic image perturbations, MAITE provides a standardized interface for evaluating model performance
across a set of test conditions. Using these tools together enables modular, reproducible assessments of AI robustness
under simulated operational risks.

The following notebooks showcase how NRTK perturbations can be applied to simulate key operational risks within a
testing and evaluation (T&E) workflow. Each notebook illustrates potential impact on model performance, utilizing MAITE
as an evaluation harness.

Demonstrating Extreme Illumination Perturbations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simulate brightness changes and evaluate model responses under lighting variability.
`View notebook <examples/maite/nrtk_brightness_perturber_demo.html>`__.

Demonstrating Visual Focus Perturbations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Apply blur and focus distortions to test performance degradation from defocus.
`View notebook <examples/maite/nrtk_focus_perturber_demo.html>`__.

Demonstrating Fog or Haze Perturbations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluate model robustness under haze-like visibility conditions using synthetic perturbations.
`View notebook <examples/maite/nrtk_haze_perturber_demo.html>`__.

Demonstrating Lens Flare Perturbations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simulate a lens flare effect on an image and analyze its average and worst case effects on model precision.
`View notebook <examples/maite/nrtk_lens_flare_demo.html>`__.

Demonstrating Resolution and Noise Transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Explore how camera-specific transformations affect model inputs and predictions.
`View notebook <examples/maite/nrtk_sensor_transformation_demo.html>`__.

Demonstrating Random Translation Perturbations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Introduce pixel-level translations and observe model sensitivity to spatial shifts.
`View notebook <examples/maite/nrtk_translation_perturber_demo.html>`__.

Demonstrating Atmospheric Turbulence Perturbations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simulate atmospheric distortion effects and assess their impact on image quality and model inference.
`View notebook <examples/maite/nrtk_turbulence_perturber_demo.html>`__.

Demonstrating Rain/Water Droplet Perturbations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simulate a rain/water droplet effect and analyze its impact on model inputs and predictions.
`View notebook <examples/maite/nrtk_water_droplet_perturber_demo.html>`__.

Demonstrating Radial Distortion Perturbations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simulate a radial distortion effect and analyze its impact on model inputs and predictions.
`View notebook <examples/maite/nrtk_radial_distortion_perturber_demo.html>`__.

Combining Perturbations with Saliency Maps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Integrate NRTK perturbations with saliency map generation to visualize how image changes affect model interpretation.
`View notebook <examples/maite/jatic-perturbations-saliency.html>`__.
