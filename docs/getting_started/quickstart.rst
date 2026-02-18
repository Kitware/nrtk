Getting Started
===============

The Natural Robustness Toolkit (NRTK) helps you test AI model robustness by
simulating real-world operational conditions. Quickly install, run, and link
your scenario to the right tools.

----

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: 🚀 Get Started in 2 Minutes
      :columns: 12 12 8 8

      **1. Install NRTK**

      .. tab-set::

         .. tab-item:: Basic Installation

            .. code-block:: bash

               pip install nrtk

         .. tab-item:: Advanced Installation

            .. code-block:: bash

               pip install nrtk[dev1,dev2,...]

            See :ref:`perturber-dependencies` for optional dev extras.

      **2. Run a Sample Perturbation**

      .. code-block:: python

         from nrtk.impls.perturb_image.environment import HazePerturber
         img_out = HazePerturber(img_in)

      See example outputs in our :doc:`Visual Perturbation Gallery </explanations/risk_factors>`.

   .. grid-item-card:: 🗺️ Map Your Risk to the Right Tool
      :columns: 12 12 4 4
      :class-card: sd-border-2

      Have a specific operational condition or test scenario?

      Use the Interactive Operational Risk Matrix to discover which perturbations represent your mission environment and conditions.

      +++

      .. button-ref:: /explanations/risk_factors
         :color: primary
         :expand:

         ➔ Interactive Risk Matrix

.. toctree::
   :hidden:

   installation
   first_perturbation
   /examples/nrtk_tutorial
   where_to_go_next
