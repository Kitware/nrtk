Applying an Operational Risk Perturbation
=========================================

Real-world deployments expose AI models to environmental and sensor-level
degradations that rarely appear in training data. Validating robustness to
these conditions is a core part of Artificial Intelligence Test & Evaluation
(AI T&E).

In this guide, you'll use NRTK's
:class:`JitterPerturber <nrtk.impls.perturb_image.optical.otf.JitterPerturber>`
to apply a physics-based **jitter perturbation** that simulates high-frequency
vibration — the kind of sensor blur caused by wind, vehicle movement, or
mechanical instability on mounted cameras. By the end, you'll be able to:

* Understand what an **operational risk perturbation** looks like in practice
* See how a few lines of code can **simulate a real-world degradation**
* Get a feel for NRTK's perturbation workflow before exploring more perturbers

When to Use This
----------------

* You want to simulate **platform vibration** from wind, vehicle movement, or
  mechanical instability.
* You need a **physics-based perturbation** that models realistic motion blur
  effects.
* You're doing early **screening of robustness** to vibration before running
  heavier T&E analysis (see the full T&E Simulation Guide →
  :doc:`JitterPerturber T&E guide </examples/maite/nrtk_jitter_perturber_demo>`).
* You want to test model performance at **reduced effective resolution** due to
  motion.

Example: Jitter Perturbation
----------------------------

JitterPerturber requires the ``pybsm`` extra. If you haven't already, install it:

.. code-block:: bash

    pip install nrtk[pybsm]

The following example loads an image, applies a jitter perturbation, and saves
the result:

.. code-block:: python

    from nrtk.impls.perturb_image.optical.otf import JitterPerturber
    import numpy as np
    from PIL import Image

    # Load your image
    image = np.array(Image.open("your_image.jpg"))

    # Apply jitter perturbation
    # img_gsd = ground sample distance (meters/pixel) for your sensor
    perturber = JitterPerturber(s_x=8e-6, s_y=8e-6)
    perturbed_image, _ = perturber(image=image, img_gsd=0.03)

    # Save the result
    Image.fromarray(perturbed_image).save("perturbed_output.jpg")

The ``s_x`` and ``s_y`` parameters control jitter amplitude — see
:ref:`key-parameters-jitter` below for details and visual comparisons.

Here's what the perturbation looks like in practice:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Baseline Image (No Jitter)
     - Image with Simulated Sensor Jitter
   * - .. image:: ../images/input.jpg
          :width: 300px
     - .. image:: ../images/output-jitter.jpg
          :width: 300px

Notice how the jitter perturbation introduces motion blur that reduces edge
sharpness and suppresses fine detail. In deployed systems, this kind of degradation
can increase missed detections, reduce confidence scores, and lower mAP — precisely
the kinds of robustness gaps T&E aims to surface before fielding. The severity of
this degradation is controlled by the jitter amplitude parameters, explored below.

.. _key-parameters-jitter:

Key Parameters
--------------

* ``s_x``, ``s_y`` — Root-mean-squared jitter amplitudes in x and y directions
  (radians). These represent the standard deviation of angular positional error
  due to platform vibration. Larger values produce a stronger Gaussian blur.

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

For a deeper dive into optical perturber parameters and OTF modeling, see the
:doc:`Optical Perturbers notebook </examples/optical_perturbers>`.

Other Operational Risks
-----------------------

NRTK provides perturbers for a range of real-world degradations beyond
vibration. Each module below includes a description, code example, parameter
comparison, and links to further resources.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: 🌫️ Haze
      :link: /explanations/operational_risk_modules/haze
      :link-type: doc

      Simulates reduced visibility from fog, mist, or light snow.

   .. grid-item-card:: 🌀 Atmospheric Turbulence
      :link: /explanations/operational_risk_modules/atmospheric_turbulence
      :link-type: doc

      Simulates blur and distortion from unstable air masses.

   .. grid-item-card:: ☀️ Extreme Illumination
      :link: /explanations/operational_risk_modules/extreme_illumination
      :link-type: doc

      Simulates under- or overexposure from harsh lighting conditions.

   .. grid-item-card:: 🔍 Target Out of Focus
      :link: /explanations/operational_risk_modules/defocus
      :link-type: doc

      Simulates optical defocus from autofocus failures or focus errors.

   .. grid-item-card:: 🔭 Radial Distortion
      :link: /explanations/operational_risk_modules/radial_distortion
      :link-type: doc

      Simulates barrel or pincushion distortion from wide-angle lenses.

   .. grid-item-card:: 📡 Sensor Noise & Resolution
      :link: /explanations/operational_risk_modules/sensor_noise_resolution
      :link-type: doc

      Simulates thermal noise, electronic noise, and resolution loss.

   .. grid-item-card:: 💧 Water Droplets
      :link: /explanations/operational_risk_modules/water_droplets
      :link-type: doc

      Simulates lens contamination from rain or water spray.

References
----------

* **API Reference:**
  :class:`JitterPerturber <nrtk.impls.perturb_image.optical.otf.JitterPerturber>`
* **Full Explanation:**
  :doc:`High-Frequency Vibration Module </explanations/operational_risk_modules/high_frequency_vibration>`
  — Physics background, OTF modeling details, and limitations
* **T&E Simulation Guide:**
  :doc:`JitterPerturber T&E guide </examples/maite/nrtk_jitter_perturber_demo>`
  — Detailed analysis, validation, datasets, and recommended parameter sweeps
* **End-to-End Tutorial:**
  :doc:`NRTK End-to-End Overview </examples/nrtk_tutorial>`
  — Complete workflow covering perturbation, factories, and model evaluation
* **Concepts:** :doc:`/explanations/nrtk_explanation`
  — Conceptual guide to NRTK's architecture and approach
* **Risk Matrix:** :doc:`/explanations/risk_factors`
  — Map real-world operational risks to NRTK perturbations
