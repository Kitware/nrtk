Target Out of Focus Simulation Module
=====================================

Optical defocus occurs when the sensor's focus plane does not align with the target distance,
causing blur that reduces image sharpness and detail. This can result from autofocus failures,
manual focus errors, rapid target movement, or fixed-focus systems operating outside their
optimal range. Defocus degrades detection performance by reducing edge contrast and obscuring
fine features.

From an ML T&E perspective, defocus matters because it introduces a form of resolution loss
that many models trained on sharp imagery are not robust to. Unlike motion blur, which is
directional, defocus blur is radially symmetric and uniformly affects all
details at the target distance. This makes it a distinct degradation mode that should be
tested separately.

NRTK's DefocusOTFPerturber models optical defocus using the Optical Transfer Function (OTF)
to simulate the characteristic blur pattern of an out-of-focus optical system. This provides
a physics-based approach to testing model sensitivity to focus errors during robustness
screening.

.. figure:: /images/risks/out-of-focus.png
  :width: 300px

Defocus Perturbation
--------------------

Simulates optical defocus effects so you can test how well a model handles blur from
incorrect focus settings or autofocus failures.

Use This When...
----------------

* You want to simulate **focus errors** in imagery from optical systems.
* You need a **physics-based perturbation** that models realistic defocus blur.
* You're doing early **screening of robustness** to focus-related degradation.
* You want to test model performance when **targets are at unexpected distances**.

Minimal Code Example
--------------------

.. code-block:: python

   from nrtk.impls.perturb_image.optical.defocus_otf_perturber import DefocusOTFPerturber

   perturber = DefocusOTFPerturber(
       w_x=5.0e-6,  # blur spot radius in x direction (m)
       w_y=5.0e-6,  # blur spot radius in y direction (m)
   )
   perturbed_img, boxes = perturber(image=img_in, boxes=boxes, img_gsd=0.03)

Key Parameters
--------------

* ``w_x`` / ``w_y`` - The 1/e blur spot radii in x and y directions (meters). These control
  the amount of defocus blur applied to the image. Higher values produce more blur.

  .. list-table::
     :widths: 33 33 33
     :header-rows: 1

     * - Light (2.0e-4)
       - Medium (5.0e-4)
       - Heavy (1.0e-3)
     * - .. image:: /images/operational_risk_modules/defocus_light.png
            :width: 200px
       - .. image:: /images/operational_risk_modules/defocus_medium.png
            :width: 200px
       - .. image:: /images/operational_risk_modules/defocus_heavy.png
            :width: 200px

**Advanced Parameters** (defaults work for most cases):

* ``interp`` - Whether to interpolate atmosphere data - Default: True.
* Sensor and scenario parameters can be passed via ``kwargs`` to customize the optical
  system being simulated.

Limitations and Next Steps
---------------------------

* API Reference:
  :class:`DefocusOTFPerturber<nrtk.impls.perturb_image.optical.defocus_otf_perturber.DefocusOTFPerturber>`
* Models **uniform defocus** across the image; does not simulate depth-dependent focus
  effects or bokeh patterns from specific aperture shapes.
* Requires ``img_gsd`` parameter when calling ``perturb()`` to properly scale effects.
* Requires optional dependencies: install via ``pip install nrtk[pybsm]``.
* See :doc:`/validation_and_trust` for cross-perturber validation status.
* Related Risks: :ref:`turbulence`, :ref:`noise-and-resolution`
