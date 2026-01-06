###############
Implementations
###############

------------------
Image Perturbation
------------------

Photometric Perturbers
^^^^^^^^^^^^^^^^^^^^^^

Photometric perturbers modify the visual appearance of images by adjusting color, brightness, contrast, sharpness,
blur, and noise properties.

.. autosummary::
   :toctree: _implementations
   :template: custom-module-template.rst
   :recursive:

   ~nrtk.impls.perturb.photometric.blur
   ~nrtk.impls.perturb.photometric.enhance
   ~nrtk.impls.perturb.photometric.noise

Geometric Perturbers
^^^^^^^^^^^^^^^^^^^^

Geometric perturbers alter the spatial positioning and orientation of images through transformations such as rotation,
scaling, cropping, and translation.

.. autosummary::
   :toctree: _implementations
   :template: custom-module-template.rst
   :recursive:

   ~nrtk.impls.perturb.geometric.random_crop_perturber
   ~nrtk.impls.perturb.geometric.random_rotation_perturber
   ~nrtk.impls.perturb.geometric.random_scale_perturber
   ~nrtk.impls.perturb.geometric.random_translation_perturber

Environment Perturbers
^^^^^^^^^^^^^^^^^^^^^^

Environment perturbers simulate atmospheric and weather-related effects that occur in real-world imaging conditions.

.. autosummary::
   :toctree: _implementations
   :template: custom-module-template.rst
   :recursive:

   ~nrtk.impls.perturb.environment.haze_perturber
   ~nrtk.impls.perturb.environment.water_droplet_perturber

Optical Perturbers
^^^^^^^^^^^^^^^^^^

Optical perturbers simulate physics-based sensor and optical effects.

.. autosummary::
   :toctree: _implementations
   :template: custom-module-template.rst
   :recursive:

   ~nrtk.impls.perturb.optical.pybsm_perturber
   ~nrtk.impls.perturb.optical.pybsm_otf_perturber
   ~nrtk.impls.perturb.optical.circular_aperture_otf_perturber
   ~nrtk.impls.perturb.optical.defocus_otf_perturber
   ~nrtk.impls.perturb.optical.detector_otf_perturber
   ~nrtk.impls.perturb.optical.jitter_otf_perturber
   ~nrtk.impls.perturb.optical.turbulence_aperture_otf_perturber
   ~nrtk.impls.perturb.optical.radial_distortion_perturber

Generative Perturbers
^^^^^^^^^^^^^^^^^^^^^

Generative perturbers use AI models to transform images through learned representations.

.. autosummary::
   :toctree: _implementations
   :template: custom-module-template.rst
   :recursive:

   ~nrtk.impls.perturb.generative.diffusion_perturber

Wrapper Perturbers
^^^^^^^^^^^^^^^^^^

Wrapper perturbers enable composition of multiple perturbations or provide integration with third-party augmentation
libraries.

.. autosummary::
   :toctree: _implementations
   :template: custom-module-template.rst
   :recursive:

   ~nrtk.impls.perturb.wrapper.albumentations_perturber
   ~nrtk.impls.perturb.wrapper.compose_perturber

Utility Components
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _implementations
   :template: custom-module-template.rst
   :recursive:

   ~nrtk.impls.utils.utils

---------------------
Perturbation Factory
---------------------

.. autosummary::
   :toctree: _implementations
   :template: custom-module-template.rst
   :recursive:

   ~nrtk.impls.perturb_image_factory.generic.linspace
   ~nrtk.impls.perturb_image_factory.generic.one_step
   ~nrtk.impls.perturb_image_factory.generic.step
   ~nrtk.impls.perturb_image_factory.generic.multivariate
