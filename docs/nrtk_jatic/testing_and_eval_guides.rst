Testing & Evaluation (T&E) Guides
=================================

The following Testing & Evaluation (T&E) guides demonstrate how NRTK-provided perturbations can be applied to test
imagery and their impact on model performance measured via the `MAITE <https://mit-ll-ai-technology.github.io/maite/>`_
evaluation workflow. Each guide is provided in the form of a notebook and follows a similar structure that demonstrates
how a T&E engineer might configure an end-to-end natural robustness evaluation to assess a particular operational risk.

Below are a list of notebooks with a short description:

* `nrtk_brightness_perturber_demo <../examples/maite/nrtk_brightness_perturber_demo.html>`_ modifies illumination
  conditions using ``NRTK's`` :ref:`BrightnessPerturber`.
* `nrtk_focus_perturber_demo <../examples/maite/nrtk_focus_perturber_demo.html>`_ modifies visual focus using
  ``NRTK's`` :ref:`DefocusOTFPerturber`.
* `nrtk_haze_perturber_demo <../examples/maite/nrtk_haze_perturber_demo.html>`_ simulates fog or haze using
  ``NRTK's`` :ref:`HazePerturber`.
* `nrtk_lens_flare_demo <../examples/maite/nrtk_lens_flare_demo.html>`_ simulates lens flares using
  ``NRTK's`` :ref:`AlbumentationsPerturber`.
* `nrtk_sensor_transformation_demo <../examples/maite/nrtk_sensor_transformation_demo.html>`_ modifies the resolution
  and noise of a sensor using ``NRTK's`` :ref:`PybsmPerturber`.
* `nrtk_translation_perturber_demo <../examples/maite/nrtk_translation_perturber_demo.html>`_ performs random
  translations of the image using ``NRTK's`` :ref:`RandomTranslationPerturber`.
* `nrtk_turbulence_perturber_demo <../examples/maite/nrtk_turbulence_perturber_demo.html>`_ modifies atmospheric
  turbulence using ``NRTK's`` :ref:`TurbulenceApertureOTFPerturber`.
* `nrtk_water_droplet_perturber_demo <../examples/maite/nrtk_water_droplet_perturber_demo.html>`_ simulates water
  droplets on a camera lens using ``NRTK's`` :ref:`WaterDropletPerturber`.
