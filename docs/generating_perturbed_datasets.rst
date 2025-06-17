Generate Perturbed Datasets
===========================

When data collects are infeasible, NRTK can be used to extend existing datasets by applying perturbations to simulate
key operational risks.

NRTK provides multiple interfaces to accomplish this task for object detection datasets, detailed below. Note that
these interfaces require COCO-format annotations. However, the library call avoids this requirement as annotations are
provided via MAITE protocols.

Command-line Interface
----------------------

  .. click:: nrtk.interop.maite.utils.bin.nrtk_perturber_cli:nrtk_perturber_cli
      :prog: nrtk-perturber
      :nested: full

NRTK-as-a-Service (NRTKaaS)
---------------------------

See `NRTKaaS documentation here <maite/nrtk_as_a_service.html>`_.

Containerized (AUKUS Data)
--------------------------

See `container documentation here <containers/aukus.html>`_.

Library Call for MAITE-compliant Data
-------------------------------------

.. automodule:: nrtk.interop.maite.utils.nrtk_perturber
    :members:
    :special-members:
