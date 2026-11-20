============
Explanations
============

Understand the problem NRTK addresses, how the toolkit is structured, and how that
structure connects to real-world operational risk.

----

.. grid:: 1
   :gutter: 3
   :padding: 2 2 0 0

   .. grid-item-card:: :material-regular:`menu_book` 1. Concepts of Robustness
      :link: /explanations/robustness_concepts
      :link-type: doc

      **The problem.** Why physics-based perturbations matter: real-world
      distribution shifts challenge computer vision models, and standard
      augmentation libraries fall short.

   .. grid-item-card:: :material-regular:`warning` 2. Operational Risk Factors
      :link: /explanations/risk_factors
      :link-type: doc
      :class-card: sd-border-2

      **The application.** How operational risk is modeled: real-world conditions
      (e.g. extreme illumination, vibration, weather) map to perturbations,
      parameter ranges, and severity levels within a structured risk matrix.

   .. grid-item-card:: :material-regular:`movie` 3. Image Perturbations on FMV
      :link: /explanations/framewise_perturber
      :link-type: doc

      **The temporal context.** When to apply image perturbations frame by frame,
      when to use a video-aware perturber to model effects that evolve over time,
      and how random state affects successive frames.

.. toctree::
   :hidden:

   robustness_concepts
   risk_factors
   framewise_perturber
