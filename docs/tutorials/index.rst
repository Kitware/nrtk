=========
Tutorials
=========

Ready to go deeper? Use these guided, end-to-end walkthroughs to learn NRTK by doing.
These tutorials assume you've completed the steps in the
:doc:`Advanced Installation </getting_started/installation>` and
:doc:`Getting Started </getting_started/quickstart>` sections.

Start Here
----------

.. grid:: 1
   :gutter: 3
   :padding: 2 2 0 0

   .. grid-item-card:: NRTK End-to-End Overview
      :class-card: sd-border-2
      :columns: 12

      It is highly recommended to start with this tutorial. This tutorial walks through NRTK's
      three core capabilities—image perturbation, perturbation factories for parameter sweeps,
      and model evaluation—in a single end-to-end notebook.

      +++

      .. button-ref:: /examples/end_to_end_overview
         :color: primary
         :outline:

         Start the Tutorial →

Continue Learning
------------------

.. grid:: 1 2 2 2
   :gutter: 3
   :padding: 2 2 0 0

   .. grid-item-card:: :material-regular:`science` T&E with MAITE
      :link: /tutorials/maite_testing_and_evaluation_guides
      :link-type: doc

      Apply perturbations within standardized T&E workflows covering
      photometric, geometric, environmental, and optical risks.

   .. grid-item-card:: :material-regular:`visibility` NRTK + Saliency
      :link: /tutorials/xaitk_saliency_analysis
      :link-type: doc

      Combine perturbations with XAITK saliency maps to understand
      model behavior under degradation.

   .. grid-item-card:: :material-regular:`event` Webinars
      :link: /tutorials/webinars/index
      :link-type: doc

      Recordings and hands-on notebooks from NRTK webinar sessions.


.. seealso::
   Choose perturbations for your deployment conditions using the
   :doc:`Operational Risk Guide </explanations/risk_factors>`.
   For individual tasks, see the :doc:`How-To Guides </how_to_guides/index>`;
   for class parameters and behavior, see the :doc:`API Implementations </reference/api/implementations>`.

.. toctree::
   :hidden:

   /examples/end_to_end_overview
   maite_testing_and_evaluation_guides
   xaitk_saliency_analysis
   webinars/index
