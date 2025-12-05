=============
NRTK Overview
=============

NRTK consists of three main parts:

Image Perturbation
------------------

The core of NRTK is based on image :term:`perturbation`. NRTK offers a wide variety of ways to perturb images
and transform bounding boxes. The perturbation classes take an image and perform a transformation based on input
parameters. :doc:`Perturbers </reference/api/implementations>` implement the
:doc:`PerturbImage </reference/api/interfaces>` interface.

Perturbation Factories
----------------------

Building upon image perturbation, :term:`perturbation factories <perturber factory>` are able to take a range of
values for parameter(s) and perform multiple perturbations on the same image. This allows for quick and simple
generation of multiple perturbations.  :doc:`Perturbation Factories </reference/api/implementations>`
implement the :doc:`PerturbImageFactory </reference/api/interfaces>` interface.

Model Evaluation
----------------

NRTK provides functionality for evaluating models in the image classification and object
detection tasks. The package also provides test orchestration functionality for performing evaluations over a sweep
of parameters in order to test model response to varying severity of image degradation. While NRTK perturbations can
be used with any evaluation harness, built-in
:doc:`NRTK Generators </reference/api/implementations>` implement the
:doc:`GenerateObjectDetectorBlackboxResponse </reference/api/interfaces>`
interface.
