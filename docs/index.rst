.. nrtk documentation master file, created by
   sphinx-quickstart on Thu Dec  1 10:21:11 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NRTK's documentation!
================================

Welcome to the documentation for the Natural Robustness Toolkit (NRTK), a platform created for
developers seeking to rigorously evaluate and enhance the robustness of computer vision models.
This toolkit simulates a wide range of real-world perturbations, focusing on sensor-specific
variables such as changes in camera focal length and aperture diameter. It provides a detailed
analysis of how these factors affect algorithm performance and expand existing datasets. Whether
you're dealing with subtle shifts in optical settings or more pronounced environmental changes,
this toolkit gives you the insights and capabilities necessary to ensure your innovative computer
vision solutions are resilient and reliable under diverse conditions.

This documentation is structured to provide you with straightforward and practical instructions and
examples, so that you can effectively leverage the toolkit to enhance the robustness and
reliability of your computer vision applications in facing real-world challenges.

Documentation Contents:
=======================

.. toctree::
   :maxdepth: 1
   :caption: Quick Start

   installation
   algorithm_list
   getting_started

.. toctree::
   :maxdepth: 1
   :caption: Tutorial

   examples/nrtk_tutorial.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Explanation

   introduction

.. toctree::
   :maxdepth: 1
   :caption: How-To

   otf_examples
   review_process
   releasing
   creating_public_release_request

.. toctree::
   :maxdepth: 1
   :caption: Reference

   interfaces
   implementations
   jatic_interoperability
   glossary
   release_notes
   style_sheet



Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
