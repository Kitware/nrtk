Introduction
============

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

Background
----------

Computer vision models are sensitive to data distribution shifts, either due to synthetic image
perturbations [1] or those naturally occurring in real data [2].
Existing image augmentation libraries (e.g. `imgaug <https://github.com/aleju/imgaug>`_ and
`albumentations <https://github.com/albumentations-team/albumentations>`_) do not cover
physics-based, sensor-specific perturbations that are relevant to operational data.

The gold standard for evaluating AI model robustness is to score against comprehensive real-test
data spanning all dimensions of expected deployment-condition variability. While there is no
substitute for collecting diverse test data, complete coverage is prohibitively expensive or
impossible in many applications. You can use NRTK to force-multiply your finite test dataset
in a principled way that covers the wider range of natural variations and corruptions expected
during real-world deployment.

.. figure:: figures/intro-fig-01.png

   Figure 1: Extending Robustness Testing with NRTK.

Use Cases
---------

You can use NRTK to assess the robustness of computer vision models trained on satellite images
to changes in different sensor parameters (e.g. focal length, aperture, pixel pitch, etc.). For
example, one use case is the task of designing a new satellite to support AI-based detection and
classification of particular objects. With NRTK, you can start with high-resolution aerial
imagery with known ground-sample-distance, then emulate imagery that would have been collected
from a hypothetical telescope with prescribed properties to explore the trade-off of telescope
cost versus performance. You can also develop validated sets of sensor perturbation parameters
and expanded datasets for comprehensive model test and evaluation (T&E).

Refer to the Getting Started section for an in-depth example of how to use NRTK.

Toolkit Overview
-----------------

The nrtk package is an open source toolkit for evaluating the natural robustness of computer
vision algorithms to various perturbations, including sensor-specific changes to camera focal
length, aperture diameter, etc. Functionality is provided through `Strategy <https://en.wikipedia
.org/wiki/Strategy_pattern>`_ and `Adapter <https://en.wikipedia.org/wiki/Adapter_pattern>`_
patterns to allow for modular integration into systems and applications.

The toolkit includes several types of general image perturbations, as well as sensor-based
perturbations based on the open source library PyBSM [3]. pyBSM rigorously models radiative
transfer and imaging-sensor physics, allowing a user to provide source images, ideally with
minimal corruptions to start. These source images are then rendered to emulate pre-sensor,
in-sensor, and post-sensor corruptions that would have been incurred by another sensor with precise
specification (altitude, atmospheric turbulence, focal length, aperture size, focus blur, pixel
pitch, quantum efficiency, shot/readout noise, and compression, among many others).

The nrtk package provides image perturbation followed by score generation and can work with any
computer vision model in a black-box manner. The perturbations themselves are independent or
agnostic of a downstream task, but nrtk's interfaces allow for evaluation of classification and
detection models.

The nrtk algorithms can also be organized according to their respective tasks:

- Image perturbation:
    * :ref:`Image Perturbation <Image Perturbation>`
    * :ref:`Perturbation Factory <Perturbation Factory>`

- Score generation:
    * :ref:`Scoring <Scoring>`
    * :ref:`End-to-End Generation and Scoring <End-to-End Generation and Scoring>`


References
----------

1. Hendrycks, Dan, and Thomas Dietterich. "Benchmarking Neural Network Robustness to Common
Corruptions and Perturbations." International Conference on Learning Representations. 2018.

2. Recht, Benjamin, et al. "Do imagenet classifiers generalize to imagenet?." International
Conference on machine learning. PMLR, 2019.

3. LeMaster, Daniel A., and Michael T. Eismann. 2017. "pyBSM: A Python package for modeling imaging
systems." Proceedings of the SPIE 10204.
