##########
Interfaces
##########

The NRTK API consists of object-oriented functor interfaces for image perturbation.
The ``PerturbImage`` interface defines the contract for applying perturbations to images,
and the ``PerturbImageFactory`` interface defines the contract for easily varying specified
parameters of a particular perturber.

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
