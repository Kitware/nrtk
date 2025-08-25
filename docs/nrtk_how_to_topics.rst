
How-To Guides
=============

This section provides task-specific guides demonstrating how to use NRTK to apply perturbations, assess model
robustness, and visualize system behavior. These examples are organized into general-purpose applications.

Each guide links to a Jupyter notebook in the ``docs/examples/`` directory of the repository.

General NRTK Examples
---------------------

.. toctree::
    :hidden:

    examples/otf_visualization.ipynb
    examples/perturbers.ipynb
    examples/albumentations_perturber.ipynb
    examples/nrtk_xaitk_workflow/image_classification_perturbation_saliency.ipynb
    .. examples/maite/augmentations.ipynb
    .. examples/maite/compute_image_metric.ipynb
    .. examples/simple_generic_generator.ipynb
    .. examples/simple_pybsm_generator.ipynb

- **Visualizing Optical Transfer Functions**

Explore and visualize different Optical Transfer Functions (OTFs) to understand their impact on image quality.
`View notebook <examples/otf_visualization.html>`__.

- **Applying Image Perturbations**

Use various image perturbation methods to simulate real-world distortions and evaluate model robustness.
`View notebook <examples/perturbers.html>`__.

- **Applying Albumentations Perturbations via NRTK**

Explore and visualize Albumentations perturbations in an NRTK context.
`View notebook <examples/albumentations_perturber.html>`__.

- **Image Classification Perturbation Saliency**

Explore how perturbations affect model predictions and how those perturbations can be interpreted using
saliency maps for the image classification task.
`View notebook <examples/nrtk_xaitk_workflow/image_classification_perturbation_saliency.html>`__.

..
    - **Visualize Item-Response Curves using NRTK** Explore and visualize the effect of varying perturbation strength
    on model prediction using NRTK components `View notebook <examples/simple_generic_generator.html>`__.

..
    - **Visualize Item-Response Curves using pyBSM via NRTK** Explore and visualize the effect of varying sensor and
    scenario perturbations on model prediction using pyBSM and NRTK components. `View notebook <examples/simple_pybsm_generator.html>`__.

Related Resources
-----------------

For broader context or foundational theory, see:

- `NRTK Tutorial <examples/nrtk_tutorial.html>`__ – Step-by-step tutorial to get started
- :doc:`nrtk_explanation` – Conceptual guide to NRTK’s architecture and approach
- :doc:`risk_factors` – Conceptual guide to understand how NRTK's perturbations map to real-world risk factors
