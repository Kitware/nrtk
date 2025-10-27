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

Enable NRTK Perturbations as MAITE Augmentations
------------------------------------------------

For tutorials utilizing these adapters to evaluate model robustness against key operational risks, see our
`Testing & Evaluation Guides <../testing_and_evaluation_notebooks.html>`_.

.. autosummary::
   :toctree: _implementations/interop
   :template: custom-module-template.rst
   :recursive:

   nrtk.interop.maite.augmentations
   nrtk.interop.maite.datasets
   nrtk.interop.maite.metadata
