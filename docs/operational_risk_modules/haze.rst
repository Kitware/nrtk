Haze Perturbation
=================

Simulates reduced visibility (fog, mist, light snow) so you can test how well a model handles low-contrast
outdoor scenes.

Use This When...
----------------

* You want to **degrade contrast** and partially occlude targets in ground/sea imagery.
* You need a **basic-install perturbation** (no pyBSM / optical dependencies).
* You're doing early **screening of robustness** to visibility loss before running heavier T&E analysis.

Minimal Code Example
--------------------

.. code-block:: python

   from nrtk.impls.perturb_image.haze_perturber import HazePerturber

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
     * - .. image:: ../images/operational_risk_modules/haze_light.png
            :width: 200px
       - .. image:: ../images/operational_risk_modules/haze_medium.png
            :width: 200px
       - .. image:: ../images/operational_risk_modules/haze_heavy.png
            :width: 200px

* ``depth_map`` - Optional 2D array describing distance from camera - Default: uniform depth map (all values = 1.0)
  for simple scenes.
* ``sky_color`` - Optional RGB triplet to approximate sky/background color - Default: estimated from the average image
  color.

Limitations and Next Steps
---------------------------

* API Reference: :class:`HazePerturber <nrtk.impls.perturb_image.generic.haze_perturber.HazePerturber>`
* Approximates **atmospheric scattering** only; it does not model full physics or wavelength-dependent effects.
  For more detailed analysis, validation details, datasets, and recommended parameter sweeps, see the
  `HazePerturber <../examples/maite/nrtk_haze_perturber_demo.html>`__ T&E guide.
* See *Validation & Trust* for cross-perturber validation status.
* Related Risks: :ref:`extreme-illumination`, :ref:`lens-water-droplet`
