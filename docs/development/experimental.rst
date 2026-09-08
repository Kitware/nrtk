============
Experimental
============

Some NRTK features are **experimental**: still under active development with APIs
that can change or disappear without a deprecation warning. Experimental
features are opt-in, so you cannot import them until you enable them.

Enabling Experimental Features
==============================

To enable experimental features, import ``nrtk.experimental`` before importing
anything experimental:

.. code-block:: python

    import nrtk.experimental  # enable in-development features

That single import will enable **all** experimental features. After opting into
experimental features, experimental features can be imported from their
usual/stable locations:

.. pytestmark: skip

.. code-block:: python

    from nrtk.interfaces import SomeExperimentalInterface
    from nrtk.impls.<subpackage> import SomeExperimentalImpl

If you try to use an experimental feature without this opt-in import, you will get an
``ImportError``, instructing you how to enable experimental features.

.. note::
   Enabling experimental features does not install anything. If an experimental feature
   needs an extra, you will still be required to install it (e.g. ``pip install nrtk[<extra>]``),
   the same as any other NRTK feature.

To summarize, up to two steps may be required to use an experimental feature:

1. Opt in to globally enable **all** of NRTK's experimental features.
2. Install any extras that are required for the desired experimental feature.

Current Experimental Features
=============================

Features under active development.

Interfaces
----------

.. autoclass:: nrtk.interfaces.PerturbVideo
   :members:

.. autoclass:: nrtk.interfaces.VideoFrame
   :members:

Implementations
---------------

.. autoclass:: nrtk.impls.perturb_video.FramewisePerturber
   :members:

   .. seealso::
      :doc:`/examples/maite/framewise_video`

   .. seealso::
      :doc:`/development/framewise_perturber`

.. autoclass:: nrtk.impls.perturb_video.optical.TurbulenceVideoPerturber
   :members:

.. autoclass:: nrtk.impls.perturb_video.CodecMacroblockPerturber
   :members:

.. autoclass:: nrtk.impls.perturb_video.burn_in.MISBST1909BurnInPerturber
   :members:

Interoperability
----------------

.. autoclass:: nrtk.interop.MAITEMultiobjectTrackingAugmentation
   :members:
