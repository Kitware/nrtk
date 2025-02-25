.. _nrtk_brightness_perturber_demo: https://gitlab.jatic.net/jatic/kitware/nrtk-jatic/-/blob/main/examples/nrtk_brightness_perturber_demo.ipynb?ref_type=heads
.. _BrightnessPerturber: https://jatic.pages.jatic.net/kitware/nrtk/_implementations/nrtk.impls.perturb_image.generic.PIL.enhance.BrightnessPerturber.html#nrtk.impls.perturb_image.generic.PIL.enhance.BrightnessPerturber
.. _nrtk_focus_perturber_demo: https://gitlab.jatic.net/jatic/kitware/nrtk-jatic/-/blob/main/examples/nrtk_focus_perturber_demo.ipynb?ref_type=heads
.. _DefocusOTFPerturber: https://jatic.pages.jatic.net/kitware/nrtk/_implementations/nrtk.impls.perturb_image.pybsm.defocus_otf_perturber.html#module-nrtk.impls.perturb_image.pybsm.defocus_otf_perturber
.. _nrtk_translation_perturber_demo: https://gitlab.jatic.net/jatic/kitware/nrtk-jatic/-/blob/main/examples/nrtk_translation_perturber_demo.ipynb?ref_type=heads
.. _RandomTranslationPerturber: https://jatic.pages.jatic.net/kitware/nrtk/_implementations/nrtk.impls.perturb_image.generic.translation_perturber.html#module-nrtk.impls.perturb_image.generic.translation_perturber
.. _nrtk_turbulence_perturber_demo: https://gitlab.jatic.net/jatic/kitware/nrtk-jatic/-/blob/main/examples/nrtk_focus_perturber_demo.ipynb?ref_type=heads
.. _TurbulenceApertureOTFPerturber: https://jatic.pages.jatic.net/kitware/nrtk/_implementations/nrtk.impls.perturb_image.pybsm.turbulence_aperture_otf_perturber.html#module-nrtk.impls.perturb_image.pybsm.turbulence_aperture_otf_perturber

Testing & Evaluation (T&E) Guides
=================================

The following Testing & Evaluation (T&E) guides demonstrate how NRTK-provided perturbations can be applied to test
imagery and their impact on model performance measured via the `MAITE <https://mit-ll-ai-technology.github.io/maite/>`_
evaluation workflow. Each guide is provided in the form of a notebook and follows a similar structure that demonstrates
how a T&E engineer might configure an end-to-end natural robustness evaluation to assess a particular operational risk.

Below are a list of notebooks with a short description:

* `nrtk_brightness_perturber_demo`_ modifies illumination conditions using ``NRTK's`` `BrightnessPerturber`_.
* `nrtk_focus_perturber_demo`_ modifies visual focus using ``NRTK's`` `DefocusOTFPerturber`_.
* `nrtk_translation_perturber_demo`_ performs random translations of the image using ``NRTK's``
  `RandomTranslationPerturber`_.
* `nrtk_turbulence_perturber_demo`_ modifies atmospheric turbulence using ``NRTK's`` `TurbulenceApertureOTFPerturber`_.




