Radial Distortion Simulation Module
====================================

Radial distortion is a common optical aberration in wide-angle and fisheye lenses that causes
straight lines to appear curved, particularly near the image edges. This distortion comes in
two forms: barrel (fisheye) distortion where lines bow outward, and pincushion distortion
where lines bow inward. These effects alter object shapes and positions in ways that can
confuse detection and tracking algorithms trained on rectilinear imagery.

From an ML T&E perspective, radial distortion matters because it introduces geometric
transformations that change object appearance based on position in the frame. Objects near
the edges experience more distortion than those at the center, creating position-dependent
variations that models may not generalize to. This is particularly relevant for systems
using wide-angle cameras for an increased field of view.

NRTK's RadialDistortionPerturber applies configurable radial distortion using a polynomial
model with three coefficients. This provides an efficient way to test model robustness to
lens distortion effects across a range of severities and distortion types during T&E.

.. figure:: /images/risks/radio-distortion.png
  :width: 300px

Radial Distortion Perturbation
------------------------------

Simulates lens distortion effects (barrel/fisheye or pincushion) so you can test how well
a model handles geometric warping from wide-angle or uncorrected optics.

Use This When...
----------------

* You want to simulate **lens distortion** from wide-angle or fisheye cameras.
* You need to test **geometric robustness** without requiring physics-based optical simulation.
* You're doing early **screening of robustness** to lens nonlinearities.
* You want to test model performance with **uncorrected camera optics**.

Minimal Code Example
--------------------

.. code-block:: python

   from nrtk.impls.perturb_image.optical.radial_distortion_perturber import RadialDistortionPerturber

   # Fisheye (barrel) distortion - positive k values
   perturber = RadialDistortionPerturber(k=[0.3, 0.0, 0.0])
   perturbed_img, boxes = perturber(image=img_in, boxes=boxes)

   # Pincushion distortion - negative k values
   perturber = RadialDistortionPerturber(k=[-0.3, 0.0, 0.0])
   perturbed_img, boxes = perturber(image=img_in, boxes=boxes)

Key Parameters
--------------

* ``k`` - List of three distortion coefficients [k1, k2, k3]. The first coefficient (k1)
  has the strongest effect. Positive values create fisheye/barrel distortion; negative
  values create pincushion distortion.

**Fisheye (Barrel) Distortion** - Positive k values push edges outward:

  .. list-table::
     :widths: 33 33 33
     :header-rows: 1

     * - Light ([0.1, 0, 0])
       - Medium ([0.3, 0, 0])
       - Heavy ([0.6, 0, 0])
     * - .. image:: /images/operational_risk_modules/radial_distortion_fisheye_light.png
            :width: 200px
       - .. image:: /images/operational_risk_modules/radial_distortion_fisheye_medium.png
            :width: 200px
       - .. image:: /images/operational_risk_modules/radial_distortion_fisheye_heavy.png
            :width: 200px

**Pincushion Distortion** - Negative k values pull edges inward:

  .. list-table::
     :widths: 33 33 33
     :header-rows: 1

     * - Light ([-0.1, 0, 0])
       - Medium ([-0.15, 0, 0])
       - Heavy ([-0.2, 0, 0])
     * - .. image:: /images/operational_risk_modules/radial_distortion_pincushion_light.png
            :width: 200px
       - .. image:: /images/operational_risk_modules/radial_distortion_pincushion_medium.png
            :width: 200px
       - .. image:: /images/operational_risk_modules/radial_distortion_pincushion_heavy.png
            :width: 200px

**Additional Parameters**:

* ``color_fill`` - RGB values for background fill where distortion creates empty regions -
  Default: [0, 0, 0] (black).

Limitations and Next Steps
---------------------------

* API Reference:
  :class:`RadialDistortionPerturber<nrtk.impls.perturb_image.optical.radial_distortion_perturber.RadialDistortionPerturber>`
* Uses a **polynomial distortion model**; does not simulate other lens aberrations like
  chromatic aberration or vignetting.
* Bounding boxes are automatically adjusted to account for the geometric transformation.
* No additional dependencies required beyond base NRTK installation.
* See :doc:`/validation_and_trust` for cross-perturber validation status.
