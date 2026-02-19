=============
How-To Guides
=============

Know what you want to accomplish? These task-focused guides show you how to complete
specific goals—such as generating perturbed datasets or running sweeps—without
revisiting background concepts.

----

General Image Perturbations
----------------------------

Guides for modifying image quality, generating perturbations from text prompts,
and wrapping external augmentation libraries.

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
         :expand:

         Open Guide →

   .. grid-item-card:: :material-regular:`extension` Albumentations Perturber
      :class-card: sd-border-1

      Use the Albumentations library through NRTK's perturber interface.

      +++

      .. button-ref:: /examples/albumentations_perturber
         :color: primary
         :outline:
         :expand:

         Open Guide →

   .. grid-item-card:: :material-regular:`auto_fix_high` Generative Perturbers
      :class-card: sd-border-1

      Use pre-trained diffusion models to synthesize realistic image
      modifications from text prompts.

      +++

      .. button-ref:: /examples/generative_perturbers
         :color: primary
         :outline:
         :expand:

         Open Guide →

Sensor & Physics Simulation
----------------------------

Guides for configuring and using pyBSM’s physics-based sensor models,
including optical transfer functions, sensor parameters, and scenario setup.

.. grid:: 1 2 2 2
   :gutter: 3
   :padding: 2 2 0 0

   .. grid-item-card:: :material-regular:`camera` Optical Perturbers
      :class-card: sd-border-1

      Simulate physics-based sensor and optical effects.

      +++

      .. button-ref:: /examples/optical_perturbers
         :color: primary
         :outline:
         :expand:

         Open Guide →

   .. grid-item-card:: :material-regular:`settings` PyBSM Default Configuration
      :class-card: sd-border-1

      Set up and customize pyBSM sensor and scenario configurations.

      +++

      .. button-ref:: /examples/pybsm_default_config
         :color: primary
         :outline:
         :expand:

         Open Guide →

.. toctree::
   :hidden:

   /examples/photometric_perturbers
   /examples/albumentations_perturber
   /examples/generative_perturbers
   /examples/optical_perturbers
   /examples/pybsm_default_config
