Extreme Illumination Simulation Module
======================================

Extreme lighting conditions, from harsh sunlight to deep shadows or nighttime operation, can
cause images to be overexposed, underexposed, or lacking in contrast. These effects lead to
missed detections, reduced confidence scores, and degraded feature extraction, particularly
for targets with limited texture or low dynamic range. Because many computer vision models
are trained on well-lit, balanced imagery, extreme illumination represents a common source
of performance degradation in outdoor and uncontrolled environments.

From an ML T&E perspective, extreme illumination matters because it introduces a form of
domain shift: the scene's pixel intensity distribution changes in ways the model has
not learned to generalize to. This makes the risk both frequent in real operations (dawn,
dusk, bright midday) and highly impactful to detection and classification performance.

NRTK's :class:`~nrtk.impls.perturb_image.photometric.enhance.BrightnessPerturber`
simulates this risk by uniformly adjusting image brightness
to approximate extreme lighting conditions. While this does not model full radiometric
effects, it provides an efficient way to probe model sensitivity to illumination variations
during early-stage robustness screening.

.. list-table::
   :widths: 50 50

   * - .. figure:: /images/risks/illumination-1.jpg
          :width: 100%
     - .. figure:: /images/risks/illumination-2.jpg
          :width: 100%

Brightness Perturbation
-----------------------

Simulates extreme illumination conditions by adjusting image brightness so you can test
how well a model handles underexposed or overexposed imagery.

Use This When...
----------------

* You want to simulate **extreme low light** (dusk, dawn, shadows) or **extreme bright light**
  (midday sun, reflections).
* You need a **simple, fast perturbation** that doesn't require physics-based sensor modeling.
* You're performing early **screening of robustness** to illumination variations before running more
  intensive T&E analysis (see the full T&E Simulation Guide →
  :doc:`BrightnessPerturber T&E guide </examples/maite/extreme_illumination>`).
* You want to test model performance under **reduced dynamic range** and contrast.

Minimal Code Example
--------------------

.. pytestmark: pillow
.. code-block:: python

   from nrtk.impls.perturb_image.photometric.enhance import BrightnessPerturber
   import numpy as np
   from PIL import Image

   # Load your image
   INPUT_IMG_FILE = 'docs/images/input.jpg'
   image = np.array(Image.open(INPUT_IMG_FILE))

   # Low brightness example
   perturber = BrightnessPerturber(factor=0.15)  # very dark
   perturbed_img, _ = perturber(image=image)

   # High brightness example
   perturber = BrightnessPerturber(factor=3.5)  # very bright
   perturbed_img, _ = perturber(image=image)

Key Parameters
--------------

* ``factor`` - Brightness enhancement factor (1.0 = original, <1.0 = darker, >1.0 = brighter).

**Low Brightness (Underexposure):**

  .. list-table::
     :widths: 33 33 33
     :header-rows: 1

     * - Light (0.85)
       - Medium (0.5)
       - Heavy (0.15)
     * - .. image:: /images/operational_risk_modules/brightness_low_light.png
            :width: 200px
       - .. image:: /images/operational_risk_modules/brightness_low_medium.png
            :width: 200px
       - .. image:: /images/operational_risk_modules/brightness_low_heavy.png
            :width: 200px

**High Brightness (Overexposure):**

  .. list-table::
     :widths: 33 33 33
     :header-rows: 1

     * - Light (1.5)
       - Medium (2)
       - Heavy (3.5)
     * - .. image:: /images/operational_risk_modules/brightness_high_light.png
            :width: 200px
       - .. image:: /images/operational_risk_modules/brightness_high_medium.png
            :width: 200px
       - .. image:: /images/operational_risk_modules/brightness_high_heavy.png
            :width: 200px

Limitations and Next Steps
---------------------------

* API Reference:
  :class:`BrightnessPerturber <nrtk.impls.perturb_image.photometric.enhance.BrightnessPerturber>`
* Applies **uniform brightness adjustment** across the entire image; does not model spatially
  varying illumination, shadows, or glare. For more detailed analysis, validation details,
  datasets, and recommended parameter sweeps, see the
  :doc:`BrightnessPerturber T&E guide </examples/maite/extreme_illumination>`.
* See :doc:`/validation_and_trust` for cross-perturber validation status.
* Related Risks: :ref:`Shadows <shadows>`, :ref:`Night Mode <night-mode>`
