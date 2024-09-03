###############
Implementations
###############

------------------
Image Perturbation
------------------

Class: AverageBlurPerturber
---------------------------
.. autoclass:: nrtk.impls.perturb_image.generic.cv2.blur.AverageBlurPerturber
   :members:
   :special-members:

Class: BrightnessPerturber
--------------------------
.. autoclass:: nrtk.impls.perturb_image.generic.PIL.enhance.BrightnessPerturber
   :members:
   :special-members:

Class: ColorPerturber
---------------------
.. autoclass:: nrtk.impls.perturb_image.generic.PIL.enhance.ColorPerturber
   :members:
   :special-members:

Class: ContrastPerturber
------------------------
.. autoclass:: nrtk.impls.perturb_image.generic.PIL.enhance.ContrastPerturber
   :members:
   :special-members:

Class: GaussianBlurPerturber
----------------------------
.. autoclass:: nrtk.impls.perturb_image.generic.cv2.blur.GaussianBlurPerturber
   :members:
   :special-members:

Class: GaussianNoisePerturber
-----------------------------
.. autoclass:: nrtk.impls.perturb_image.generic.skimage.random_noise.GaussianNoisePerturber
   :members:
   :special-members:

Class: MedianBlurPerturber
--------------------------
.. autoclass:: nrtk.impls.perturb_image.generic.cv2.blur.MedianBlurPerturber
   :members:
   :special-members:

Class: nop_perturber
--------------------
.. autoclass:: nrtk.impls.perturb_image.generic.nop_perturber.NOPPerturber
   :members:
   :special-members:

Class: PepperNoisePerturber
---------------------------
.. autoclass:: nrtk.impls.perturb_image.generic.skimage.random_noise.PepperNoisePerturber
   :members:
   :special-members:

Class: PybsmPerturber
---------------------
.. autoclass:: nrtk.impls.perturb_image.pybsm.perturber.PybsmPerturber
   :members:
   :special-members:

Class: PybsmScenario
--------------------
.. autoclass:: nrtk.impls.perturb_image.pybsm.scenario.PybsmScenario
   :members:
   :special-members:

Class: PybsmSensor
------------------
.. autoclass:: nrtk.impls.perturb_image.pybsm.sensor.PybsmSensor
   :members:
   :special-members:

Class: CircleApertureOTFPerturber
---------------------------------
.. autoclass:: nrtk.impls.perturb_image.pybsm.circular_aperture_otf_perturber.CircularApertureOTFPerturber
   :members:
   :special-members:
   
Class: DetectorOTFPerturber
-------------------------
.. autoclass:: nrtk.impls.perturb_image.pybsm.detector_otf_perturber.DetectorOTFPerturber
   :members:
   :special-members:

Class: JitterOTFPerturber
-------------------------
.. autoclass:: nrtk.impls.perturb_image.pybsm.jitter_otf_perturber.JitterOTFPerturber
   :members:
   :special-members:

Class: SaltAndPepperNoisePerturber
----------------------------------
.. autoclass:: nrtk.impls.perturb_image.generic.skimage.random_noise.SaltAndPepperNoisePerturber
   :members:
   :special-members:

Class: SaltNoisePerturber
-------------------------
.. autoclass:: nrtk.impls.perturb_image.generic.skimage.random_noise.SaltNoisePerturber
   :members:
   :special-members:

Class: SharpnessPerturber
-------------------------
.. autoclass:: nrtk.impls.perturb_image.generic.PIL.enhance.SharpnessPerturber
   :members:
   :special-members:

Class: SpeckleNoisePerturber
----------------------------
.. autoclass:: nrtk.impls.perturb_image.generic.skimage.random_noise.SpeckleNoisePerturber
   :members:
   :special-members:


--------------------
Perturbation Factory
--------------------

Class: CustomPybsmPerturbImageFactory
-------------------------------------
.. autoclass:: nrtk.impls.perturb_image_factory.pybsm.CustomPybsmPerturbImageFactory
   :members:
   :special-members:

Class: StepPerturbImageFactory
------------------------------
.. autoclass:: nrtk.impls.perturb_image_factory.generic.step.StepPerturbImageFactory
   :members:
   :special-members:

-------------
Image Metrics
-------------

Class: NIIRSImageMetric
---------------------
.. autoclass:: nrtk.impls.image_metric.niirs_image_metric.NIIRSImageMetric
   :members:
   :special-members:

Class: SNRImageMetric
---------------------
.. autoclass:: nrtk.impls.image_metric.snr_image_metric.SNRImageMetric
   :members:
   :special-members:

-------
Scoring
-------

Class: ClassAgnosticPixelwiseIoUScorer
--------------------------------------
.. autoclass:: nrtk.impls.score_detections.class_agnostic_pixelwise_iou_scorer.ClassAgnosticPixelwiseIoUScorer
   :members:
   :special-members:

Class: COCOScorer
-----------------
.. autoclass:: nrtk.impls.score_detections.coco_scorer.COCOScorer
   :members:
   :special-members:

Class: NOPScorer
----------------
.. autoclass:: nrtk.impls.score_detections.nop_scorer.NOPScorer
   :members:
   :special-members:

Class: RandomScorer
-------------------
.. autoclass:: nrtk.impls.score_detections.random_scorer.RandomScorer
   :members:
   :special-members:

---------------------------------
End-to-End Generation and Scoring
---------------------------------

Class: SimpleGenericGenerator
-----------------------------
.. autoclass:: nrtk.impls.gen_object_detector_blackbox_response.simple_generic_generator.SimpleGenericGenerator
   :members:
   :special-members:

Class: SimplePybsmGenerator
---------------------------
.. autoclass:: nrtk.impls.gen_object_detector_blackbox_response.simple_pybsm_generator.SimplePybsmGenerator
   :members:
   :special-members:
