import numpy as np
import pytest
from pybsm.otf import darkCurrentFromDensity
from PIL import Image
from contextlib import nullcontext as does_not_raise
from typing import Any, Dict, ContextManager, Tuple

from smqtk_core.configuration import configuration_test_helper

from nrtk.impls.perturb_image.pybsm.perturber import PybsmPerturber
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario

from ..test_perturber_utils import pybsm_perturber_assertions

INPUT_IMG_FILE = './examples/pybsm/data/M-41 Walker Bulldog (USA) width 319cm height 272cm.tiff'
EXPECTED_IMG_FILE = './tests/impls/perturb_image/pybsm/data/Expected Output.tiff'


class TestPyBSMPerturber:
    def createSampleSensorandScenario(self) -> Tuple[PybsmSensor, PybsmScenario]:

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
        eta = 0.4  # guess

        # detector width is assumed to be equal to the pitch
        wx = p
        wy = p
        # integration time (s) - this is a maximum, the actual integration time will be
        # determined by the well fill percentage
        intTime = 30.0e-3

        # dark current density of 1 nA/cm2 guess, guess mid range for a silicon camera
        darkCurrent = darkCurrentFromDensity(1e-5, wx, wy)

        # rms read noise (rms electrons)
        readNoise = 25.0

        # maximum ADC level (electrons)
        maxN = 96000.0

        # bit depth
        bitdepth = 11.9

        # maximum allowable well fill (see the paper for the logic behind this)
        maxWellFill = .6

        # jitter (radians) - The Olson paper says that its "good" so we'll guess 1/4 ifov rms
        sx = 0.25*p/f
        sy = sx

        # drift (radians/s) - again, we'll guess that it's really good
        dax = 100e-6
        day = dax

        # etector quantum efficiency as a function of wavelength (microns)
        # for a generic high quality back-illuminated silicon array
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

    def test_consistency(self) -> None:
        """
        Run on a dummy image to ensure output matches precomputed results.
        """
        image = np.array(Image.open(INPUT_IMG_FILE))
        expected = np.array(Image.open(EXPECTED_IMG_FILE))
        img_gsd = 3.19/160.0
        sensor, scenario = self.createSampleSensorandScenario()
        # Test perturb interface directly
        inst = PybsmPerturber(sensor=sensor, scenario=scenario, groundRange=10000)
        pybsm_perturber_assertions(perturb=inst.perturb, image=image, expected=expected,
                                   additional_params={'img_gsd': img_gsd})

        # Test callable
        pybsm_perturber_assertions(
            perturb=PybsmPerturber(sensor=sensor, scenario=scenario, groundRange=10000),
            image=image,
            expected=expected,
            additional_params={'img_gsd': img_gsd}
        )

    @pytest.mark.parametrize("param_name, param_value", [
        ('groundRange', 10000),
        ('groundRange', 20000),
        ('groundRange', 30000),
        ('altitude', 10000),
        ('ihaze', 2)
    ])
    def test_reproducibility(self, param_name: str, param_value: Any) -> None:
        """
        Ensure results are reproducible.
        """
        # Test perturb interface directly
        image = np.array(Image.open(INPUT_IMG_FILE))
        sensor, scenario = self.createSampleSensorandScenario()
        inst = PybsmPerturber(sensor=sensor, scenario=scenario, **{param_name: param_value})
        img_gsd = 3.19/160.0
        out_image = pybsm_perturber_assertions(perturb=inst.perturb, image=image, expected=None,
                                               additional_params={'img_gsd': img_gsd})
        pybsm_perturber_assertions(perturb=inst.perturb, image=image, expected=out_image,
                                   additional_params={'img_gsd': img_gsd})

    def test_configuration(self) -> None:
        """
        Test configuration stability.
        """
        sensor, scenario = self.createSampleSensorandScenario()
        inst = PybsmPerturber(sensor=sensor, scenario=scenario)
        for i in configuration_test_helper(inst):
            assert i.sensor.name == sensor.name
            assert i.sensor.D == sensor.D
            assert i.sensor.f == sensor.f
            assert i.sensor.px == sensor.px
            assert np.array_equal(i.sensor.optTransWavelengths, sensor.optTransWavelengths)
            assert np.array_equal(i.sensor.opticsTransmission, sensor.opticsTransmission)
            assert i.sensor.eta == sensor.eta
            assert i.sensor.wx == sensor.wx
            assert i.sensor.wy == sensor.wy
            assert i.sensor.intTime == sensor.intTime
            assert i.sensor.darkCurrent == sensor.darkCurrent
            assert i.sensor.readNoise == sensor.readNoise
            assert i.sensor.maxN == sensor.maxN
            assert i.sensor.bitdepth == sensor.bitdepth
            assert i.sensor.maxWellFill == sensor.maxWellFill
            assert i.sensor.sx == sensor.sx
            assert i.sensor.sy == sensor.sy
            assert i.sensor.dax == sensor.dax
            assert i.sensor.day == sensor.day
            assert np.array_equal(i.sensor.qewavelengths, sensor.qewavelengths)
            assert np.array_equal(i.sensor.qe, sensor.qe)

            assert i.scenario.name == scenario.name
            assert i.scenario.ihaze == scenario.ihaze
            assert i.scenario.altitude == scenario.altitude
            assert i.scenario.groundRange == scenario.groundRange
            assert i.scenario.aircraftSpeed == scenario.aircraftSpeed
            assert i.scenario.targetReflectance == scenario.targetReflectance
            assert i.scenario.targetTemperature == scenario.targetTemperature
            assert i.scenario.backgroundReflectance == scenario.backgroundReflectance
            assert i.scenario.backgroundTemperature == scenario.backgroundTemperature
            assert i.scenario.haWindspeed == scenario.haWindspeed
            assert i.scenario.cn2at1m == scenario.cn2at1m

            assert np.array_equal(i.reflectance_range, inst.reflectance_range)

    @pytest.mark.parametrize("reflectance_range, expectation", [
        (np.array([.05, .5]), does_not_raise()),
        (np.array([.01, .5]), does_not_raise()),
        (np.array([.05]),
            pytest.raises(ValueError, match=r"Reflectance range array must have length of 2")),
        (np.array([.5, .05]),
            pytest.raises(ValueError, match=r"Reflectance range array values must be strictly ascending"))
    ])
    def test_configuration_bounds(self, reflectance_range: np.ndarray, expectation: ContextManager) -> None:
        """
        Test that an exception is properly raised (or not) based on argument value.
        """
        sensor, scenario = self.createSampleSensorandScenario()
        with expectation:
            PybsmPerturber(sensor=sensor, scenario=scenario, reflectance_range=reflectance_range)

    @pytest.mark.parametrize("additional_params, expectation", [
        ({"img_gsd": 3.19/160.}, does_not_raise()),
        ({}, pytest.raises(ValueError, match=r"'img_gsd' must be present in image metadata for this perturber"))
    ])
    def test_additional_params(self, additional_params: Dict[str, Any], expectation: ContextManager) -> None:
        """
        Test variations of additional params.
        """
        sensor, scenario = self.createSampleSensorandScenario()
        perturber = PybsmPerturber(sensor=sensor, scenario=scenario, reflectance_range=np.array([.05, .5]))
        image = np.array(Image.open(INPUT_IMG_FILE))
        with expectation:
            _ = perturber(image, additional_params)
