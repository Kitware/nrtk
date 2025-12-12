##########
Interfaces
##########

The NRTK API consists of a number of object-oriented functor interfaces for item-response curve
(IRC) generation,
namely for assessing model response to perturbations on given input data. These interfaces focus on black-box IRC
generation. In addition to the driver, or *generator*, of this task, there are two other main components: reference
image perturbation in preparation for black-box testing and black-box scoring for model outputs. The
``PeturbImageFactory`` interface provides the utility to easily vary specified parameters of a particular perturber.
The generator will execute upon a given perturber factory as well as a given model and scorer to generate the IRC for
given input data. We define a few similar interfaces for performing the IRC generation, separated by the intermediate
algorithmic use cases, one for object detection and one for image classification.


We explicitly do not require an abstraction for the black-box operations to fit inside. This is intended to allow for
applications using these interfaces while leveraging existing functionality, which only need to perform data formatting
to fit the input defined here. Note, however, some interfaces are defined for certain black-box concepts as part of the
SMQTK ecosystem (e.g. in `SMQTK-Classifier <https://github.com/Kitware/SMQTK-Classifier>`_,
`SMQTK-Detection
<https://github.com/Kitware/SMQTK-Detection>`_, and other SMQTK-* modules).


These interfaces are based on the plugin and configuration features provided by
`SMQTK-Core <https://github.com/Kitware/SMQTK-Core>`_, to allow convenient hooks into implementation, discoverability,
and factory generation from runtime configuration. This allows for both opaque discovery of interface implementations
from a class-method on the interface class object, as well as instantiation of a concrete instance via a JSON-like
configuration fed in from an outside resource.

.. figure:: /figures/api-docs-fig-01.svg

   Figure 1: Abstract Interface Inheritance.

.. When adding new classes within interfaces, sort them alphabetically.

------------------
Image Perturbation
------------------

Interface: PerturbImage
-----------------------
.. autoclass:: nrtk.interfaces.perturb_image.PerturbImage
   :members:
   :special-members:

--------------------
Perturbation Factory
--------------------

Interface: PerturbImageFactory
------------------------------
.. autoclass:: nrtk.interfaces.perturb_image_factory.PerturbImageFactory
   :members:
   :special-members:

-------------
Image Metrics
-------------

Interface: ImageMetric
----------------------
.. autoclass:: nrtk.interfaces.image_metric.ImageMetric
   :members:
   :special-members:

-------
Scoring
-------

Interface: ScoreDetections
--------------------------
.. autoclass:: nrtk.interfaces.score_detections.ScoreDetections
   :members:
   :special-members:

---------------------------------
End-to-End Generation and Scoring
---------------------------------

Interface: GenerateObjectDetectorBlackboxResponse
-------------------------------------------------
.. autoclass:: nrtk.interfaces.gen_object_detector_blackbox_response.GenerateObjectDetectorBlackboxResponse
   :members:
   :special-members:
