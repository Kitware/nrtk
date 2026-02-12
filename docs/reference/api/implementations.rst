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

Blur
""""

Blur perturbers apply various blurring effects using cv2.

.. autosummary::
   :toctree: _implementations
   :template: custom-class-template.rst
   :nosignatures:

   ~nrtk.impls.perturb_image.photometric.blur.AverageBlurPerturber
   ~nrtk.impls.perturb_image.photometric.blur.GaussianBlurPerturber
   ~nrtk.impls.perturb_image.photometric.blur.MedianBlurPerturber

Enhance
"""""""

Enhancement perturbers adjust image properties using PIL.

.. autosummary::
   :toctree: _implementations
   :template: custom-class-template.rst
   :nosignatures:

   ~nrtk.impls.perturb_image.photometric.enhance.BrightnessPerturber
   ~nrtk.impls.perturb_image.photometric.enhance.ColorPerturber
   ~nrtk.impls.perturb_image.photometric.enhance.ContrastPerturber
   ~nrtk.impls.perturb_image.photometric.enhance.SharpnessPerturber

Noise
"""""

Noise perturbers add random noise patterns using skimage.

.. autosummary::
   :toctree: _implementations
   :template: custom-class-template.rst
   :nosignatures:

   ~nrtk.impls.perturb_image.photometric.noise.GaussianNoisePerturber
   ~nrtk.impls.perturb_image.photometric.noise.PepperNoisePerturber
   ~nrtk.impls.perturb_image.photometric.noise.SaltAndPepperNoisePerturber
   ~nrtk.impls.perturb_image.photometric.noise.SaltNoisePerturber
   ~nrtk.impls.perturb_image.photometric.noise.SpeckleNoisePerturber

Geometric Perturbers
^^^^^^^^^^^^^^^^^^^^

Geometric perturbers alter the spatial positioning and orientation of images through transformations such as rotation,
scaling, cropping, and translation.

.. autosummary::
   :toctree: _implementations
   :template: custom-class-template.rst
   :nosignatures:

   ~nrtk.impls.perturb_image.geometric.random.RandomCropPerturber
   ~nrtk.impls.perturb_image.geometric.random.RandomRotationPerturber
   ~nrtk.impls.perturb_image.geometric.random.RandomScalePerturber
   ~nrtk.impls.perturb_image.geometric.random.RandomTranslationPerturber

Environment Perturbers
^^^^^^^^^^^^^^^^^^^^^^

Environment perturbers simulate atmospheric and weather-related effects that occur in real-world imaging conditions.

.. autosummary::
   :toctree: _implementations
   :template: custom-class-template.rst
   :nosignatures:

   ~nrtk.impls.perturb_image.environment.HazePerturber
   ~nrtk.impls.perturb_image.environment.WaterDropletPerturber

Optical Perturbers
^^^^^^^^^^^^^^^^^^

Optical perturbers simulate physics-based sensor and optical effects.

.. autosummary::
   :toctree: _implementations
   :template: custom-class-template.rst
   :nosignatures:

   ~nrtk.impls.perturb_image.optical.otf.CircularAperturePerturber
   ~nrtk.impls.perturb_image.optical.otf.DefocusPerturber
   ~nrtk.impls.perturb_image.optical.otf.DetectorPerturber
   ~nrtk.impls.perturb_image.optical.otf.JitterPerturber
   ~nrtk.impls.perturb_image.optical.PybsmPerturber
   ~nrtk.impls.perturb_image.optical.radial_distortion_perturber.RadialDistortionPerturber
   ~nrtk.impls.perturb_image.optical.otf.TurbulenceAperturePerturber

Generative Perturbers
^^^^^^^^^^^^^^^^^^^^^

Generative perturbers use AI models to transform images through learned representations.

.. autosummary::
   :toctree: _implementations
   :template: custom-class-template.rst
   :nosignatures:

   ~nrtk.impls.perturb_image.generative.DiffusionPerturber

Utility Perturbers
^^^^^^^^^^^^^^^^^^

Utility perturbers enable composition of multiple perturbations or provide integration with third-party augmentation
libraries.

.. autosummary::
   :toctree: _implementations
   :template: custom-class-template.rst
   :nosignatures:

   ~nrtk.impls.perturb_image.AlbumentationsPerturber
   ~nrtk.impls.perturb_image.ComposePerturber

Utility Components
^^^^^^^^^^^^^^^^^^

Utility functions that support perturbation operations.

.. autosummary::
   :toctree: _implementations
   :template: custom-function-template.rst
   :nosignatures:

   ~nrtk.impls.perturb_image.optical.otf.load_default_config

---------------------
Perturbation Factory
---------------------

Perturbation factories generate collections of perturbers with varying parameter values, enabling systematic
exploration of perturbation parameter spaces.

.. autosummary::
   :toctree: _implementations
   :template: custom-class-template.rst
   :nosignatures:

   ~nrtk.impls.perturb_image_factory.PerturberLinspaceFactory
   ~nrtk.impls.perturb_image_factory.PerturberMultivariateFactory
   ~nrtk.impls.perturb_image_factory.PerturberOneStepFactory
   ~nrtk.impls.perturb_image_factory.PerturberStepFactory
