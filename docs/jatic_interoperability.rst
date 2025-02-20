######################
JATIC Interoperability
######################

.. autosummary::
   :toctree: _implementations/interop
   :template: custom-module-template.rst
   :recursive:

   nrtk.interop.maite.interop.object_detection
   nrtk.interop.maite.interop.image_classification

=====
Utils
=====

-----------------------------
NRTK perturber CLI Entrypoint
-----------------------------

.. click:: nrtk.interop.maite.utils.bin.nrtk_perturber_cli:nrtk_perturber_cli
    :prog: nrtk-perturber
    :nested: full

------------------------------------
Augmented MAITE dataset(s) generator
------------------------------------

.. automodule:: nrtk.interop.maite.utils.nrtk_perturber
    :members:
    :special-members:
