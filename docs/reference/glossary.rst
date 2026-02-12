========
Glossary
========

.. glossary::
   :sorted:

   augmentation
     See :term:`perturbation`.

   COCO
     Common Objects in Context (COCO) — A large-scale dataset that is a widely used benchmark for object detection and
     segmentation. The COCO format is also used in the :ref:`Interoperability` module.

   environment perturber
     Environment perturbers simulate atmospheric and weather-related effects that occur in real-world imaging
     conditions.

   generative perturber
     Generative perturbers use AI models to transform images through learned representations.

   geometric perturber
     Geometric perturbers alter the spatial positioning and orientation of images through transformations such as
     rotation, scaling, cropping, and translation.

   item response curve
     A graphical representation of how mean image scores change based on perturber values.

   MAITE
     Modular AI Trustworthy Engineering — a framework for evaluating the trustworthiness and robustness of AI systems
     using standardized metrics and workflows. View
     `MAITE documentation <https://mit-ll-ai-technology.github.io/maite/>`_ or
     :ref:`NRTK integration <Interoperability>`.

   natural robustness
     A model's ability to maintain performance despite variations or changes in the environment or inputs that are
     naturally occurring, not specifically designed for testing or manipulation.

   optical perturber
     Optical perturbers simulate physics-based sensor and optical effects.

   Optical Transfer Function (OTF)
     A mathematical model that describes how an imaging system reduces detail and sharpness in an image due to physical
     limitations such as diffraction, motion, or sensor imperfections. In NRTK, perturbing OTF parameters simulates
     various sensor and environmental effects.

   perturber
     A reusable component that defines and applies a specific type of :term:`perturbation` (e.g. haze or blur) to
     image data.

   perturber factory
     A factory method implementation for creating perturbers. Perturbers can be customized by changing thetas and
     theta keys.

   perturbation
     A modification applied to input data to simulate noise, environmental degradation, or sensor artifacts.

   photometric perturber
     Photometric perturbers modify the visual appearance of images by adjusting color, brightness, contrast, sharpness,
     blur, and noise properties.

   pyBSM
     Python Based Sensor Model (pyBSM) — A toolset for modeling and simulating physical sensor effects such as blur,
     sensor noise, and environmental conditions. View on `GitHub <https://github.com/Kitware/pybsm>`_.

   saliency
     A measure of how much influence a part of an input has on a model's output.

   sensor transformation
     A change applied to image data to simulate different sensor behaviors (e.g. wavelength response, resolution,
     distortion).

   theta key(s)
     A string (or list of strings) that is the name of the pertubrer parameter to modify. A single string is used
     for generic factories and a list of strings is used for the multivariate factory.

   thetas
     A list of values (or list of lists) containing the values for the perturber parameter. A single list is used
     for generic factories and a list of lists is used for the multivariate factory.

   utility perturber
     Utility perturbers enable composition of multiple perturbations or provide integration with third-party
     augmentation libraries.
