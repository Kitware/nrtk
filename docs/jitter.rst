
Jitter OTF
==========

Overview
--------

The Jitter OTF simulates the wideband jitter of an optical sensor's line of sight. This is approximated as a Gaussian blur effect caused by minor, rapid movements of the sensor or camera, quantified by variance in the jitter intensity.

The following code snippet shows how to implement the Jitter OTF (see :ref:`jitter`), which you can use to simulate different levels of sensor jitter and study their effects on image quality.


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
    from pybsm.otf import darkCurrentFromDensity
    from typing import Tuple
    from PIL import Image

    from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
    from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario

    def createSampleSensorandScenario() -> Tuple[PybsmSensor,
                                                        PybsmScenario]:
        name = 'L32511x'

        # telescope focal length (m)
        f = 4
        # Telescope diameter (m)
        D = 275e-3

        # detector pitch (m)
        p = .008e-3

        # Optical system transmission, red  band first (m)
        optTransWavelengths = np.array([0.58-.08, 0.58+.08])*1.0e-6
        # guess at the full system optical transmission (excluding obscuration)
        opticsTransmission = 0.5*np.ones(optTransWavelengths.shape[0])

        # Relative linear telescope obscuration
        eta = 0.4

        # detector width is assumed to be equal to the pitch
        wx = p
        wy = p
        # integration time (s) - this is a maximum, the actual integration
        # time will be determined by the well fill percentage
        intTime = 30.0e-3

        # dark current density of 1 nA/cm2, assume mid range for a
        # silicon camera
        darkCurrent = darkCurrentFromDensity(1e-5, wx, wy)

        # rms read noise (rms electrons)
        readNoise = 25.0

        # maximum ADC level (electrons)
        maxN = 96000.0

        # bit depth
        bitdepth = 11.9

        # maximum allowable well fill
        maxWellFill = .6

        # jitter (radians)
        # assume 1/4 ifov rms
        sx = 0.25*p/f
        sy = sx

        # drift (radians/s)
        dax = 100e-6
        day = dax

        # quantum efficiency as a function of wavelength (microns)
        # for a generic, high-quality, back-illuminated silicon array
        # https://www.photometrics.com/resources/learningzone/quantumefficiency.php
        qewavelengths = np.array([.3, .4, .5, .6, .7, .8, .9, 1.0, 1.1])*1.0e-6
        qe = np.array([0.05, 0.6, 0.75, 0.85, .85, .75, .5, .2, 0])

        sensor = PybsmSensor(name, D, f, p, optTransWavelengths,
                                opticsTransmission, eta, wx, wy,
                                intTime, darkCurrent, readNoise,
                                maxN, bitdepth, maxWellFill, sx, sy,
                                dax, day, qewavelengths, qe)

        altitude = 9000.0
        # range to target
        groundRange = 60000.0

        scenario_name = 'niceday'
        # weather model
        ihaze = 1
        scenario = PybsmScenario(scenario_name, ihaze, altitude, groundRange)
        scenario.aircraftSpeed = 100.0

        return sensor, scenario


    INPUT_IMG_FILE = '/home/local/your.name/Projects/CDAO/nrtk/examples/pybsm/data/M-41 Walker Bulldog (USA) width 319cm height 272cm.tiff'
    image = np.array(Image.open(INPUT_IMG_FILE))
    print(image.shape)
    sensor, scenario = createSampleSensorandScenario()
    img_gsd = 3.19/160.0

    otf = JitterOTFPerturber(sensor=sensor, scenario=scenario, name="test_name")
    out_image = otf.perturb(image, additional_params={'img_gsd': img_gsd})
    print(out_image, out_image.shape)

    out_file = "/home/local/your.name/Projects/CDAO/nrtk/tests/impls/perturb_image/pybsm/data/jitter_otf_expected_output.tiff"
    Image.fromarray(out_image).save(out_file)

    print(otf.jitOTF.shape, type(otf.jitOTF))
    print(np.save("/home/local/your.name/Projects/CDAO/nrtk/examples/pybsm/data/jitter_otf.npy", otf.jitOTF))

Resulting Image
---------------

The output image below shows the effects of the Jitter OTF on the original input. This result illustrates the Gaussian blur introduced due to simulated sensor jitter.

.. figure:: images/output.jpg

   Figure 2: Output image.
