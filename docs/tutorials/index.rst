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
      :link: /examples/nrtk_tutorial
      :link-type: doc
      :class-card: sd-border-2
      :columns: 12

      It is highly recommended to start with this tutorial if you haven't
      already. It walks through NRTK's three core capabilities—image
      perturbation, perturbation factories for parameter sweeps, and model
      evaluation—in a single end-to-end notebook.

      +++

      .. button-ref:: /examples/nrtk_tutorial
         :color: primary
         :expand:

         Start the Tutorial →

Continue Learning
------------------

.. grid:: 1 1 3 3
   :gutter: 3
   :padding: 2 2 0 0

   .. grid-item-card:: :material-regular:`science` T&E with MAITE
      :link: /tutorials/testing_and_evaluation_notebooks
      :link-type: doc

      Apply perturbations within standardized T&E workflows covering
      photometric, geometric, environmental, and optical risks.

   .. grid-item-card:: :material-regular:`image` Classification + Saliency
      :link: /examples/nrtk_xaitk_workflow/image_classification_perturbation_saliency
      :link-type: doc

      Combine perturbations with XAITK saliency maps to understand
      classification model behavior under degradation.

   .. grid-item-card:: :material-regular:`center_focus_strong` Detection + Saliency
      :link: /examples/nrtk_xaitk_workflow/object_detection_perturbation_saliency
      :link-type: doc

      Extend the saliency workflow to object detection, visualizing
      shifts in detector attention and bounding boxes.

.. toctree::
   :hidden:

   testing_and_evaluation_notebooks
   /examples/nrtk_xaitk_workflow/image_classification_perturbation_saliency
   /examples/nrtk_xaitk_workflow/object_detection_perturbation_saliency
