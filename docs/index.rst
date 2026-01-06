.. nrtk documentation master file, created by
   sphinx-quickstart on Thu Dec  1 10:21:11 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NRTK's documentation!
================================


======================================================================================================================

**The Natural Robustness Toolkit (NRTK) is an open source toolkit for generating operationally realistic perturbations
to evaluate the natural robustness of computer vision algorithms.**

.. container:: nrtk-quickstart-button-container

   .. button-ref:: getting_started/quickstart
      :color: primary
      :class: nrtk-quickstart-button

      🚀 Get Started with NRTK QuickStart Hub

======================================================================================================================

Welcome to the documentation for NRTK, a tool created for developers and Test and Evaluation
(T&E) engineers seeking to rigorously evaluate and enhance the robustness of computer vision models.
This toolkit simulates a wide range of real-world perturbations, focusing on sensor-specific
variables such as changes in camera focal length and aperture diameter. It enables detailed
analysis of how these factors affect algorithm performance and expand existing datasets. Whether
you're dealing with subtle shifts in optical settings or more pronounced environmental changes,
this toolkit gives you the insights and capabilities necessary to ensure your innovative computer
vision solutions are resilient and reliable under diverse conditions.

This documentation is structured to provide you with straightforward and practical instructions and
examples, so that you can effectively leverage the toolkit to enhance the robustness and
reliability of your computer vision applications in facing real-world challenges.

.. grid:: 1 2 2 2
   :gutter: 4
   :padding: 2 2 0 0
   :class-container: sd-text-center

   .. grid-item-card:: Getting Started
      :link: getting_started/index
      :link-type: doc

      New to NRTK? Installation, quick start, and introductory tutorials.

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

   .. grid-item-card:: Interoperability
      :link: interoperability/index
      :link-type: doc

      MAITE Adapters for T&E workflows

   .. grid-item-card:: Development
      :link: development/index
      :link-type: doc

      Contributing guidelines and development resources.

   .. grid-item-card:: Release Notes
      :link: release_notes/index
      :link-type: doc

      Version history and changelog.

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

   getting_started/index
   tutorials/index
   how_to_guides/index
   explanations/index
   reference/index
   interoperability/index
   development/index
   release_notes/index

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
