High-Frequency Vibration Simulation Module
===========================================

Vibrations in the sensor platform from wind, vehicle movement, or mechanical instability induce
jitter that blurs image details and reduces effective resolution. These effects can cause missed
detections, tracking failures, and degraded feature extraction, especially for small or
low-contrast targets. Because many computer vision models are trained on stable, crisp imagery,
vibration-induced blur represents a common source of performance degradation in real-world
deployments.

From an ML T&E perspective, high-frequency vibration matters because it introduces motion blur
that the model has not learned to generalize to. This makes the risk both frequent in operational
environments (ground vehicles, maritime platforms, handheld cameras) and highly impactful on
performance metrics such as detection mAP and tracking stability.

NRTK's JitterPerturber simulates this risk by modeling the optical transfer function (OTF)
effects of platform vibration. This physics-based approach provides an efficient way to probe
model sensitivity to motion blur during early-stage robustness screening.

.. figure:: /images/risks/jitter.png
  :width: 300px

Jitter Perturbation
-------------------

Simulates high-frequency vibration effects using physics-based OTF modeling, so you can test
how well a model handles motion blur caused by platform instability.

Use This When...
----------------

* You want to simulate **platform vibration** from wind, vehicle movement, or mechanical instability.
* You need a **physics-based perturbation** that models realistic motion blur effects.
* You're doing early **screening of robustness** to vibration before running heavier T&E analysis
  (see the full T&E
  Simulation Guide → :doc:`JitterPerturber T&E guide </examples/maite/nrtk_jitter_perturber_demo>`).
* You want to test model performance at **reduced effective resolution** due to motion.

Minimal Code Example
--------------------

.. code-block:: python

   from nrtk.impls.perturb_image.optical.otf import JitterPerturber

   perturber = JitterPerturber(
       s_x=0.0,  # no jitter in x direction
       s_y=5e-4   # medium jitter in y direction
   )
   perturbed_img, perturbed_boxes = perturber.perturb(image=img, boxes=boxes, img_gsd=0.03)

Key Parameters
--------------

* ``s_x``, ``s_y`` - Root-mean-squared jitter amplitudes in x and y directions (radians).
  These represent the standard deviation of angular positional error due to platform vibration.

  .. list-table::
     :widths: 33 33 33
     :header-rows: 1

     * - Light (0, 3e-4)
       - Medium (0, 5e-4)
       - Heavy (0, 1e-3)
     * - .. image:: /images/operational_risk_modules/jitter_light.png
            :width: 200px
       - .. image:: /images/operational_risk_modules/jitter_medium.png
            :width: 200px
       - .. image:: /images/operational_risk_modules/jitter_heavy.png
            :width: 200px

Limitations and Next Steps
---------------------------

* API Reference: :class:`JitterPerturber <nrtk.impls.perturb_image.optical.otf.JitterPerturber>`
* Models **platform jitter** via optical transfer function; does not simulate rolling shutter
  effects or complex motion patterns. For more detailed analysis, validation details, datasets,
  and recommended parameter sweeps, see the
  :doc:`JitterPerturber T&E guide </examples/maite/nrtk_jitter_perturber_demo>`.
* See :doc:`/validation_and_trust` for cross-perturber validation status.
* Related Risks: :ref:`target-out-of-focus`, :ref:`turbulence`
