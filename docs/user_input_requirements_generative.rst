######################
Generative Perturbers
######################

Generative perturbers use AI models to transform images through learned representations. These perturbations leverage
deep learning models, such as diffusion models, to apply complex, content-aware transformations that go beyond
traditional image processing techniques.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Perturbation
     - Required Inputs
     - Optional Parameters

   * - :ref:`DiffusionPerturber`
     - Image (converts to RGB)
     - * ``model_name``
       * ``prompt``
       * ``seed``
       * ``num_inference_steps``
       * ``text_guidance_scale``
       * ``image_guidance_scale``
       * ``device``
       * ``boxes``
