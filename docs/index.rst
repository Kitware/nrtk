.. nrtk documentation master file, created by
   sphinx-quickstart on Thu Dec  1 10:21:11 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NRTK's documentation!
================================


======================================================================================================================

**The Natural Robustness Toolkit (NRTK) is an open source toolkit for generating operationally realistic perturbations
to evaluate the natural robustness of computer vision algorithms.**

======================================================================================================================


Welcome to the documentation for NRTK, a tool created for developers and Test and Evaluation
(T&E) engineers seeking to rigorously evaluate and enhance the robustness of computer vision models.
This toolkit simulates a wide range of real-world perturbations, focusing on sensor-specific
variables such as changes in camera focal length and aperture diameter. It provides a detailed
analysis of how these factors affect algorithm performance and expand existing datasets. Whether
you're dealing with subtle shifts in optical settings or more pronounced environmental changes,
this toolkit gives you the insights and capabilities necessary to ensure your innovative computer
vision solutions are resilient and reliable under diverse conditions.

This documentation is structured to provide you with straightforward and practical instructions and
examples, so that you can effectively leverage the toolkit to enhance the robustness and
reliability of your computer vision applications in facing real-world challenges.

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

For T&E engineers focusing on AI model testing, NRTK provides several key functionalities:


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

Documentation Contents:
=======================

.. toctree::
   :maxdepth: 1
   :caption: Quick Start

   installation
   getting_started
   algorithm_list

.. toctree::
   :maxdepth: 1
   :caption: Explanation

   nrtk_explanation
   risk_factors

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   examples/nrtk_tutorial.ipynb

.. toctree::
   :maxdepth: 1
   :caption: How-To

   nrtk_how_to_topics
   contributing
   release_process
   miscellaneous/creating_public_release_request

.. toctree::
   :maxdepth: 1
   :caption: Reference

   interfaces
   implementations
   interoperability
   glossary
   containers
   release_notes
   miscellaneous/style_sheet



Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
