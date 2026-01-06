NRTK QuickStart Hub
===================

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

      **2. Run Your First Perturbation**

      .. code-block:: python

         from nrtk.impls.perturb_image.environment.haze_perturber import HazePerturber
         img_out = HazePerturber(img_in)

      See example outputs in our :doc:`Visual Perturbation Gallery </examples/perturbers>`.

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

----

🔍 Explore More
---------------

.. container:: explore-more-grid

   .. grid:: 1 2 4 4
      :gutter: 2

      .. grid-item-card:: 🚀 Getting Started
         :link: first_perturbation
         :link-type: doc

      .. grid-item-card:: 📚 NRTK Concepts
         :link: /explanations/nrtk_explanation
         :link-type: doc

      .. grid-item-card:: 🔌 Core API Interfaces
         :link: /reference/api/interfaces
         :link-type: doc

      .. grid-item-card:: ⚙️ NRTK Implementations
         :link: /reference/api/implementations
         :link-type: doc

      .. grid-item-card:: 📋 User Input Requirements
         :link: /reference/perturber_reference/index
         :link-type: doc

      .. grid-item-card:: ✅ Validation and Trust
         :link: validation_and_trust
         :link-type: doc

      .. grid-item-card:: 📊 Testing & Evaluation Guides
         :link: /tutorials/testing_and_evaluation_notebooks
         :link-type: doc

      .. grid-item-card:: 📟 NRTK CLI Tool
         :link: /reference/containers
         :link-type: doc

----
