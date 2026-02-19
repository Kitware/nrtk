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
      :link: /explanations/nrtk_explanation
      :link-type: doc

      **The problem.** Why physics-based perturbations matter: real-world
      distribution shifts challenge computer vision models, and standard
      augmentation libraries fall short.

   .. grid-item-card:: :material-regular:`account_tree` 2. NRTK Overview
      :link: /explanations/nrtk_overview
      :link-type: doc

      **The approach.** How NRTK’s core components—image perturbers and
      perturbation factories—are structured to bridge the Test &
      Evaluation (T&E) gap and support systematic robustness evaluation.

   .. grid-item-card:: :material-regular:`warning` 3. Operational Risk Factors
      :link: /explanations/risk_factors
      :link-type: doc
      :class-card: sd-border-2

      **The application.** How operational risk is modeled: real-world conditions
      (e.g. extreme illumination,vibration, weather) map to perturbations,
      parameter ranges, and severity levels within a structured risk matrix.

.. toctree::
   :hidden:

   nrtk_explanation
   nrtk_overview
   risk_factors
