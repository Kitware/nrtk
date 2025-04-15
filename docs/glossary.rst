========
Glossary
========

   augmentation
     See perturbation.

   COCO
     Common Objects in Context (COCO) — A large-scale dataset that is a widely used benchmark for object detection and
     segmentation. In NRTK, the COCO Scorer is a tool that evaluates object detection results using metrics from that
     benchmark. The COCO format is also used in the :ref:`Interoperability` module.

   generator
     A customizable component that takes a perturber factory, a scorer, and an object detector to generate item
     response curves.

   item response curve
     A graphical representation of how mean image scores change based on perturber values.

   MAITE
     Modular AI Trustworthy Engineering — a framework for evaluating the trustworthiness and robustness of AI systems
     using standardized metrics and workflows. View
     `MAITE documentation <https://mit-ll-ai-technology.github.io/maite/>`_ or
     :ref:`NRTK integration <Interoperability>`.

   natural robustness
     A model’s ability to maintain performance despite variations or changes in the environment or inputs that are
     naturally occurring, not specifically designed for testing or manipulation.

   Optical Transfer Function (OTF)
     A mathematical model that describes how an imaging system reduces detail and sharpness in an image due to physical
     limitations such as diffraction, motion, or sensor imperfections. In NRTK, perturbing OTF parameters simulates
     various sensor and environmental effects.

   perturber
     A reusable component that defines and applies a specific type of perturbation (e.g. haze or blur) to image data.

   perturber factory
     A factory method implementation for creating perturbers. Perturbers can be customized by changing thetas and
     theta keys.

   perturbation
     A modification applied to input data to simulate noise, environmental degradation, or sensor artifacts.

   pyBSM
     Python Based Sensor Model (pyBSM) — A toolset for modeling and simulating physical sensor effects such as blur,
     sensor noise, and environmental conditions. View on `GitHub <https://github.com/Kitware/pybsm>`_.

   saliency
     A measure of how much influence a part of an input has on a model’s output.

   sensor transformation
     A change applied to image data to simulate different sensor behaviors (e.g. wavelength response, resolution,
     distortion).

   theta key(s):
     A string (or list of strings) that is the name of the pertubrer parameter to modify. A single string is used
     for generic factories and a list of strings is used for the pyBSM factory.

   thetas:
     A list of values (or list of lists) containing the values for the perturber parameter. A single list is used
     for generic factories and a list of lists is used for the pyBSM factory.
