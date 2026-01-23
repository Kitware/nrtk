======================
JATIC Interoperability
======================

Why Interoperability Matters
----------------------------

Many robustness testing workflows benefit from using NRTK alongside other
`JATIC <https://cdao.pages.jatic.net/public/>`_ tools like
`Modular AI Trustworthy Engineering (MAITE) <https://github.com/mit-ll-ai-technology/maite>`_. While NRTK focuses on
realistic image perturbations, MAITE provides a standardized interface for evaluating model performance across a set of
test conditions. Using these tools together enables modular, reproducible assessments of AI robustness under
sensor-driven variation.

To integrate NRTK with MAITE, start here to incorporate NRTK perturbations into
MAITE workflows using MAITE-compliant augmentation, dataset, and metadata wrappers.

Enable NRTK Perturbations as MAITE Augmentations
------------------------------------------------

For tutorials utilizing these adapters to evaluate model robustness against key operational risks, see our
:doc:`Testing & Evaluation Guides </tutorials/testing_and_evaluation_notebooks>`.

.. autosummary::
   :toctree: _implementations/interop
   :template: custom-class-template.rst
   :recursive:

   ~nrtk.interop.MAITEClassificationAugmentation
   ~nrtk.interop.MAITEDetectionAugmentation
