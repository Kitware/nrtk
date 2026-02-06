.. nrtk documentation master file, created by
   sphinx-quickstart on Thu Dec  1 10:21:11 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NRTK's documentation!
================================

**Version**: |release| | **Date**: |today|

**The Natural Robustness Toolkit (NRTK) is an open source toolkit for generating operationally realistic perturbations
to evaluate the natural robustness of computer vision algorithms.**

NRTK enables developers and T&E engineers to simulate sensor-specific and environmental perturbations—such as
changes in focal length, aperture, and atmospheric conditions—to rigorously assess computer vision model robustness
without costly real-world data collection.

.. grid:: 1 2 2 2
   :gutter: 3
   :padding: 2 2 0 0
   :class-container: sd-text-center

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc

      Learn NRTK through guided end-to-end examples.

   .. grid-item-card:: How-To Guides
      :link: how_to_guides/index
      :link-type: doc

      Task-based instructions and workflow recipes.

   .. grid-item-card:: Explanations
      :link: explanations/index
      :link-type: doc

      Robustness concepts and operational risk factors in computer vision.

   .. grid-item-card:: Reference
      :link: reference/index
      :link-type: doc

      Perturbers, APIs, schemas, and implementation details.

.. note::
   New to NRTK? :doc:`Getting Started </getting_started/quickstart>` walks you through
   installation and your first operational risk perturbation.

Why NRTK?
---------

NRTK addresses the critical gap in evaluating computer vision model resilience to real-world operational conditions
beyond what traditional image augmentation libraries cover. T&E engineers need precise methods to assess how models
respond to sensor-specific variables (focal length, aperture diameter, pixel pitch) and environmental factors without
the prohibitive costs of exhaustive data collection. NRTK leverages pyBSM's physics-based models to rigorously simulate
how imaging sensors capture and process light, enabling systematic robustness testing across parameter sweeps,
identification of performance boundaries, and visualization of model degradation. This capability is particularly
valuable for satellite and aerial imaging applications, where engineers can simulate hypothetical sensor configurations
to support cost-performance trade-off analysis during system design—ensuring AI models maintain reliability when
deployed on actual hardware facing natural perturbations in the field.

Testing & Evaluation Tasks
--------------------------

For T&E engineers focusing on AI model testing, NRTK enables several key functionalities:


* **Robustness Testing**: Evaluating model performance when inputs are perturbed or under distribution shift
  (e.g., new environments, camera angles).

* **Model Performance Evaluation**: Utilizing metrics like precision, recall, mAP (mean Average Precision), and IoU
  (Intersection over Union) specifically for object detection tasks.

* **Edge Case Testing**:  Identifying and testing challenging scenarios such as adverse weather conditions, low light,
  occlusions, or rare object appearances.

By incorporating NRTK into their testing processes, T&E engineers can conduct thorough assessments of AI models,
ensuring they meet robustness and reliability standards before deployment.

.. :auto acknowledgment:

Acknowledgment
--------------

This material is based upon work supported by the Chief Digital and Artificial Intelligence Office under Contract No.
519TC-23-9-2032. The views and conclusions contained herein are those of the author(s) and should not be interpreted as
necessarily representing the official policies or endorsements, either expressed or implied, of the U.S. Government.

.. :auto acknowledgment:

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started/quickstart
   tutorials/index
   how_to_guides/index
   explanations/index
   validation_and_trust
   reference/index
   interoperability/maite/jatic_interoperability
   containers/aukus
   development/index
   release_notes/index
