Sensor Noise and Resolution Effects Simulation Module
=============================================

Sensor noise and resolution degradation arise from physical limitations in imaging hardware.
Thermal noise accumulates when sensors overheat, electronic noise increases with high ISO settings
in poor lighting, and resolution suffers when pixel pitch is coarse or when targets are distant.
These effects reduce signal-to-noise ratio, erase fine details, and introduce quantization artifacts,
leading to missed detections, lower confidence scores, and fragmented tracking, especially for
small or low-texture targets.

From an ML T&E perspective, sensor noise and resolution degradation matter because they introduce
data quality issues that models rarely encounter during training. Unlike geometric or photometric
perturbations, which preserve underlying signal structure, sensor-level degradations fundamentally
alter the information available to the model. This makes these risks particularly relevant
for evaluating robustness in real-world deployments where sensor hardware may be stressed or
suboptimal.

NRTK's PybsmPerturber simulates these risks using physics-based sensor modeling (via pyBSM) that
accounts for thermal noise (dark current), electronic noise (read noise), and effective resolution
(pixel pitch). This provides a comprehensive way to probe model sensitivity to sensor-level
degradation during T&E.

Sensor Noise and Resolution Perturbation
-----------------------------------------

Simulates realistic sensor noise and resolution effects using physics-based modeling so you can
test how well a model handles thermal noise from overheating, electronic noise from high ISO,
and resolution degradation from coarse pixel pitch.

Use This When...
----------------

* You want to simulate **thermal noise** from sensor overheating or long exposures.
* You want to simulate **electronic noise** from high ISO settings or poor lighting conditions.
* You want to simulate **resolution degradation** due to coarse pixel pitch or distant targets.
* You need a **full physics-based sensor simulation** that models realistic image formation.
* You're performing **comprehensive robustness screening** at the sensor level to detect degradation
   before deployment (see the full T&E Simulation Guide →
  :doc:`PybsmPerturber T&E guide </examples/maite/nrtk_sensor_transformation_demo>`).

Minimal Code Example
--------------------

.. code-block:: python

   from nrtk.impls.perturb_image.optical import PybsmPerturber
   import numpy as np

   # Simulate moderate sensor noise (thermal + electronic)
   perturber = PybsmPerturber(
       dark_current=3e11,  # e-/s - thermal noise
       read_noise=25.0,   # e- RMS - electronic noise
       D=275e-3,          # aperture diameter (meters)
       f=4,               # focal length
       p_x=8.0e-6,        # pixel pitch x (meters)
       p_y=8.0e-6,        # pixel pitch y (meters)
       altitude=10.0,     # camera height (meters)
       ground_range=14.0, # distance to target
       opt_trans_wavelengths=np.array([0.50e-6, 0.66e-6]),
   )
   perturbed_img, perturbed_boxes = perturber.perturb(image=img, boxes=boxes, img_gsd=0.03)

Key Parameters
--------------

**Dark Current (Thermal Noise)**

* ``dark_current`` - Thermal noise from sensor heating (electrons/second, e-/s).
  Increases exponentially with temperature and accumulates during exposure. Simulates
  overheating or long integration times.

  .. list-table::
     :widths: 33 33 33
     :header-rows: 1

     * - Light (1e11 e-/s)
       - Medium (3e11 e-/s)
       - Heavy (1e12 e-/s)
     * - .. image:: /images/operational_risk_modules/sensor_dark_current_light.png
            :width: 200px
       - .. image:: /images/operational_risk_modules/sensor_dark_current_medium.png
            :width: 200px
       - .. image:: /images/operational_risk_modules/sensor_dark_current_heavy.png
            :width: 200px

**Read Noise (Electronic Noise)**

* ``read_noise`` - Electronic noise from sensor readout amplifiers (electrons RMS, e-).
  Increases with sensor gain (ISO) and is the primary source of noise in low-light or
  high-ISO imaging.

  .. list-table::
     :widths: 33 33 33
     :header-rows: 1

     * - Light (5e5 e-)
       - Medium (1e6 e-)
       - Heavy (1e7 e-)
     * - .. image:: /images/operational_risk_modules/sensor_read_noise_light.png
            :width: 200px
       - .. image:: /images/operational_risk_modules/sensor_read_noise_medium.png
            :width: 200px
       - .. image:: /images/operational_risk_modules/sensor_read_noise_heavy.png
            :width: 200px

**Pixel Pitch (Resolution)**

* ``p_x``, ``p_y`` - Detector center-to-center spacing in x and y directions (meters).
  A larger pitch means fewer pixels per unit ground distance, resulting in lower effective
  resolution. Simulates coarse sensor arrays or distant targets.

  .. list-table::
     :widths: 33 33 33
     :header-rows: 1

     * - Light / High Res (200 µm)
       - Medium (300 µm)
       - Heavy / Low Res (500 µm)
     * - .. image:: /images/operational_risk_modules/sensor_resolution_light.png
            :width: 200px
       - .. image:: /images/operational_risk_modules/sensor_resolution_medium.png
            :width: 200px
       - .. image:: /images/operational_risk_modules/sensor_resolution_heavy.png
            :width: 200px

Advanced Parameters
-------------------

These parameters have reasonable defaults but can be tuned for specific scenarios:

* ``D`` - Aperture diameter (meters). Default varies by scenario.
* ``f`` - Focal length. Default varies by scenario.
* ``int_time`` - Integration time / exposure duration (seconds). Default: 30 ms.
* ``bit_depth`` - ADC quantization resolution (bits). Default: 12 bits (4096 levels).
* ``max_n`` - Maximum electron well capacity (electrons). Default: 100,000 e-.
* ``max_well_fill`` - Maximum well fill fraction (0.0 to 1.0). Default: 1.0.
* ``altitude`` - Sensor height above ground (meters).
* ``ground_range`` - Distance from sensor to target (meters).

Limitations and Next Steps
---------------------------

* API Reference:
  :class:`PybsmPerturber <nrtk.impls.perturb_image.optical.PybsmPerturber>`
* Provides **full physics-based sensor simulation**, including noise, resolution, and optical
  effects. For more detailed analysis, validation details, datasets, and recommended parameter
  sweeps, see the
  :doc:`PybsmPerturber T&E guide </examples/maite/nrtk_sensor_transformation_demo>`.
* Requires optional dependencies: install via ``pip install nrtk[pybsm]``.
* See :doc:`/validation_and_trust` for cross-perturber validation status.
* Related Risks: :ref:`noise-and-resolution`
