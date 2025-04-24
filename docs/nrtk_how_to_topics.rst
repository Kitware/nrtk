
How-To Guides
=============

This section provides task-specific guides demonstrating how to use NRTK to apply perturbations, assess model
robustness, and visualize system behavior. These examples are organized into general-purpose applications and those
specifically integrated with Modular AI Trustworthy Engineering (MAITE) workflows.

Each guide links to a Jupyter notebook in the ``docs/examples/`` directory of the repository.

.. toctree::
   :hidden:

    examples/nrtk_tutorial.ipynb
    examples/otf_visualization.ipynb
    examples/perturbers.ipynb
    examples/coco_scorer.ipynb
    examples/simple_generic_generator.ipynb
    examples/simple_pybsm_generator.ipynb
    examples/maite/augmentations.ipynb
    examples/maite/compute_image_metric.ipynb
    examples/maite/jatic-perturbations-saliency.ipynb
    examples/maite/nrtk_brightness_perturber_demo.ipynb
    examples/maite/nrtk_focus_perturber_demo.ipynb
    examples/maite/nrtk_haze_perturber_demo.ipynb
    examples/maite/nrtk_sensor_transformation_demo.ipynb
    examples/maite/nrtk_translation_perturber_demo.ipynb
    examples/maite/nrtk_turbulence_perturber_demo.ipynb


General NRTK Examples
---------------------

- **Visualizing Optical Transfer Functions**

Explore and visualize different Optical Transfer Functions (OTFs) to understand their impact on image quality.
`View notebook <examples/otf_visualization.html>`__.


- **Applying Image Perturbations**

Use various image perturbation methods to simulate real-world distortions and evaluate model robustness.
`View notebook <examples/perturbers.html>`__.

- **Applying Albumentations Perturbations via NRTK**

Explore and visualize Albumentations perturbations in an NRTK context.
`View notebook <examples/albumentations_perturber.html>`__.

- **Evaluating Models with COCO Scoring**

Use COCO scoring to assess object detection model performance across perturbed inputs.
`View notebook <examples/coco_scorer.html>`__.

..
    - **Visualize Item-Response Curves using NRTK** Explore and visualize the effect of varying perturbation strength
    on model prediction using NRTK components `View notebook <examples/simple_generic_generator.html>`__.

..
    - **Visualize Item-Response Curves using pyBSM via NRTK** Explore and visualize the effect of varying sensor and
    scenario perturbations on model prediction using pyBSM and NRTK components. `View notebook <examples/simple_pybsm_generator.html>`__.

Testing & Evaluation Guides with MAITE
--------------------------------------

The following notebooks showcase how NRTK perturbations can be applied to simulate key operational risks within a T&E
workflow. Each notebook illustrates potential impact on model performance, utilizing MAITE as an evaluation harness.

- **Combining Perturbations with Saliency Maps**

Integrate NRTK perturbations with saliency map generation to visualize how image changes affect model interpretation.
`View notebook <examples/maite/jatic-perturbations-saliency.html>`__.

- **Demonstrating Extreme Illumination Perturbations**

Simulate brightness changes and evaluate model responses under lighting variability.
`View notebook <examples/maite/nrtk_brightness_perturber_demo.html>`__.

- **Demonstrating Visual Focus Perturbations**

Apply blur and focus distortions to test performance degradation from defocus.
`View notebook <examples/maite/nrtk_focus_perturber_demo.html>`__.

- **Demonstrating Fog or Haze Perturbations**

Evaluate model robustness under haze-like visibility conditions using synthetic perturbations.
`View notebook <examples/maite/nrtk_haze_perturber_demo.html>`__.

- **Demonstrating Lens Flare Perturbations**

Simulate lens flare effects and asses the impact on model interpretation.
`View notebook <examples/maite/nrtk_lens_flare_demo.html>`__.

- **Demonstrating Resolution and Noise Transformations**

Explore how camera-specific transformations affect model inputs and predictions.
`View notebook <examples/maite/nrtk_sensor_transformation_demo.html>`__.

- **Demonstrating Random Translation Perturbations**

Introduce pixel-level translations and observe model sensitivity to spatial shifts.
`View notebook <examples/maite/nrtk_translation_perturber_demo.html>`__.

- **Demonstrating Atmospheric Turbulence Perturbations**

Simulate atmospheric distortion effects and assess their impact on image quality and model inference.
`View notebook <examples/maite/nrtk_turbulence_perturber_demo.html>`__.

- **Demonstrating Lens Flare Perturbations**

Simulate a lens flare effect on an image and analyze its average and worst case effects on model precision.
`View notebook <examples/maite/nrtk_lens_flare_perturber_demo.html>`__.


Related Resources
-----------------

For broader context or foundational theory, see:

- `NRTK Tutorial <examples/nrtk_tutorial.html>`__ – Step-by-step tutorial to get started
- :doc:`nrtk_explanation` – Conceptual guide to NRTK’s architecture and approach
