=============
How-To Guides
=============

Know what you want to accomplish? These task-focused guides show you how to complete
specific goals—such as generating perturbed datasets or running sweeps—without
revisiting background concepts.

----

.. grid:: 1 2 2 2
   :gutter: 3
   :padding: 2 2 0 0

   .. grid-item-card:: :material-regular:`palette` Photometric Perturbers
      :class-card: sd-border-1

      Adjust color, brightness, contrast, sharpness, blur, and noise properties.

      +++

      .. button-ref:: /examples/photometric_perturbers
         :color: primary
         :outline:

         Open Guide →

   .. grid-item-card:: :material-regular:`extension` Albumentations Perturber
      :class-card: sd-border-1

      Use the Albumentations library through NRTK's perturber interface.

      +++

      .. button-ref:: /examples/albumentations_perturber
         :color: primary
         :outline:

         Open Guide →

   .. grid-item-card:: :material-regular:`camera` Optical Perturbers
      :class-card: sd-border-1

      Simulate physics-based sensor and optical effects.

      +++

      .. button-ref:: /examples/optical_perturbers
         :color: primary
         :outline:

         Open Guide →

   .. grid-item-card:: :material-regular:`auto_fix_high` Generative Perturbers *(Experimental)*
      :class-card: sd-border-1

      Use pre-trained diffusion models to synthesize realistic image
      modifications from text prompts.

      +++

      .. button-ref:: /examples/generative_perturbers
         :color: primary
         :outline:

         Open Guide →

.. seealso::
   Use the :doc:`Operational Risk Guide </explanations/risk_factors>` to select a perturbation for your scenario.
   Follow the :doc:`Tutorials </tutorials/index>` for complete evaluation workflows, and consult the
   :doc:`API Implementations </reference/api/implementations>` for class parameters and behavior.

.. toctree::
   :hidden:

   /examples/photometric_perturbers
   /examples/albumentations_perturber
   /examples/optical_perturbers
   /examples/generative_perturbers
