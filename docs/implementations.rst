###############
Implementations
###############

------------------
Image Perturbation
------------------

.. autosummary::
   :toctree: _implementations
   :template: custom-module-template.rst
   :recursive:

   nrtk.impls.perturb_image.albumentations.albumentations_perturber
   nrtk.impls.perturb_image.generic.cv2.blur
   nrtk.impls.perturb_image.generic.PIL.enhance
   nrtk.impls.perturb_image.generic.skimage.random_noise
   nrtk.impls.perturb_image.generic.nop_perturber
   nrtk.impls.perturb_image.generic.compose_perturber
   nrtk.impls.perturb_image.generic.crop_perturber
   nrtk.impls.perturb_image.generic.translation_perturber
   nrtk.impls.perturb_image.generic.haze_perturber
   nrtk.impls.perturb_image.pybsm.perturber
   nrtk.impls.perturb_image.pybsm.scenario
   nrtk.impls.perturb_image.pybsm.sensor
   nrtk.impls.perturb_image.pybsm.circular_aperture_otf_perturber
   nrtk.impls.perturb_image.pybsm.defocus_otf_perturber
   nrtk.impls.perturb_image.pybsm.detector_otf_perturber
   nrtk.impls.perturb_image.pybsm.jitter_otf_perturber
   nrtk.impls.perturb_image.pybsm.turbulence_aperture_otf_perturber

---------------------
Perturbation Factory
---------------------

.. autosummary::
   :toctree: _implementations
   :template: custom-module-template.rst
   :recursive:

   nrtk.impls.perturb_image_factory.pybsm
   nrtk.impls.perturb_image_factory.generic.linspace_step
   nrtk.impls.perturb_image_factory.generic.one_step
   nrtk.impls.perturb_image_factory.generic.step

-------------
Image Metrics
-------------

.. autosummary::
   :toctree: _implementations
   :template: custom-module-template.rst
   :recursive:

   nrtk.impls.image_metric.niirs_image_metric
   nrtk.impls.image_metric.snr_image_metric

-------
Scoring
-------

.. autosummary::
   :toctree: _implementations
   :template: custom-module-template.rst
   :recursive:

   nrtk.impls.score_detections.class_agnostic_pixelwise_iou_scorer
   nrtk.impls.score_detections.coco_scorer
   nrtk.impls.score_detections.nop_scorer
   nrtk.impls.score_detections.random_scorer

---------------------------------
End-to-End Generation and Scoring
---------------------------------

.. autosummary::
   :toctree: _implementations
   :template: custom-module-template.rst
   :recursive:

   nrtk.impls.gen_object_detector_blackbox_response.simple_generic_generator
   nrtk.impls.gen_object_detector_blackbox_response.simple_pybsm_generator
