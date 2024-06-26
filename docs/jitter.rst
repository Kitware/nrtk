
Jitter OTF
==========

Overview
--------

The Jitter OTF simulates the wideband jitter of an optical sensor's line of sight. This is approximated as a Gaussian blur effect caused by minor, rapid movements of the sensor or camera, quantified by variance in the jitter intensity.

The following code snippet shows how to implement the :ref:`JitterOTFPerturber <Class:
JitterOTFPerturber>`, which you can use to simulate different levels of sensor jitter and study
their effects on image quality. By knowing the scenario parameter (in this case, the altitude)
and adjusting the sigma_x (sx) and sigma_y (sy) parameters, you can simulate different levels of
sensor jitter and study their effects on image quality.

Input Image
-----------

Below is an example of an input image that will undergo a Jitter OTF perturbation. This image represents the initial state before any transformation.

.. figure:: images/input.jpg

   Figure 1: Input image.


Code Sample
-----------

Below is some example code that applies a Jitter OTF transformation::

    from nrtk.impls.perturb_image.pybsm.jitter_otf_perturber import JitterOTFPerturber
    import numpy as np
    from PIL import Image

    INPUT_IMG_FILE = 'docs/images/input.jpg'
    image = np.array(Image.open(INPUT_IMG_FILE))

    otf = JitterOTFPerturber(sx=8e-6, sy=8e-6, name="test_name")
    out_image = otf.perturb(image)

This code uses default values and provides a sample input image. However, you can adjust
the parameters and use your own image to visualize the perturbation. The sx and sy parameters
(the root-mean-squared jitter amplitudes in radians, in the x and y directions) are
the primary way to customize a jitter perturber. Larger jitter amplitude generate a
larger Gaussian blur kernel.

Resulting Image
---------------

The output image below shows the effects of the Jitter OTF on the original input. This result illustrates the Gaussian blur introduced due to simulated sensor jitter.

.. figure:: images/output-jitter.jpg

   Figure 2: Output image.
