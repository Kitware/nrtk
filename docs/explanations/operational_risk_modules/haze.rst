Haze Simulation Module
======================

Weather conditions reduce visibility between the sensor and target by lowering contrast,
softening edges, and obscuring fine details. These effects can cause missed detections,
reduced confidence scores, fragmented or unstable tracks, and noisier segmentation masks
— especially for small or low-texture targets. Because many outdoor computer-vision models
are implicitly trained on clear-weather imagery, visibility degradation is a common source
of performance drops and a key friction to assess in T&E.

From an ML T&E perspective, haze and mist matter because they introduce a form of domain
shift: the pixel statistics of the scene change in ways the model has not learned to
generalize to. This makes the risk both frequent in real operations and highly impactful
to performance.

NRTK's HazePerturber approximates this risk by reducing contrast and introducing
scattering-like effects guided by a depth map and estimated sky color. This does not
attempt full atmospheric physics, but it provides an efficient way to probe model
sensitivity to visibility loss during early-stage robustness screening.

.. figure:: /images/risks/mist.png
  :width: 300px

Haze Perturbation
-----------------

Simulates reduced visibility (fog, mist, light snow) so you can test how well a model
handles low-contrast outdoor scenes.

Use This When...
----------------

* You want to **degrade contrast** and partially occlude targets in ground/sea imagery.
* You need a **basic-install perturbation** (no pyBSM / optical dependencies).
* You're doing early **screening of robustness** to visibility loss before running heavier
  T&E analysis (see the full T&E Simulation Guide →
  :doc:`HazePerturber T&E guide </examples/maite/nrtk_haze_perturber_demo>`).

Minimal Code Example
--------------------

.. code-block:: python

   from nrtk.impls.perturb.environment.haze_perturber import HazePerturber

   perturber = HazePerturber(factor=1.0)  # medium haze
   img_out = perturber.perturb(img_in)

Key Parameters
--------------

* ``factor`` - Overall haze strength.

  .. list-table::
     :widths: 33 33 33
     :header-rows: 1

     * - Light (0.5)
       - Medium (1.0)
       - Heavy (1.5)
     * - .. image:: /images/operational_risk_modules/haze_light.png
            :width: 200px
       - .. image:: /images/operational_risk_modules/haze_medium.png
            :width: 200px
       - .. image:: /images/operational_risk_modules/haze_heavy.png
            :width: 200px

* ``depth_map`` - Optional 2D array describing distance from camera - Default: uniform
  depth map (all values = 1.0) for simple scenes.
* ``sky_color`` - Optional RGB triplet to approximate sky/background color -
  Default: estimated from the average image color.

Limitations and Next Steps
---------------------------

* API Reference: :class:`HazePerturber <nrtk.impls.perturb.environment.haze_perturber.HazePerturber>`
* Approximates **atmospheric scattering** only; it does not model full physics or
  wavelength-dependent effects. For more detailed analysis, validation details, datasets,
  and recommended parameter sweeps, see the
  :doc:`HazePerturber T&E guide </examples/maite/nrtk_haze_perturber_demo>`.
* See *Validation & Trust* for cross-perturber validation status.
* Related Risks: :ref:`extreme-illumination`, :ref:`lens-water-droplet`
