import numpy as np
import pytest
from pybsm.otf import dark_current_from_density
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
        opt_trans_wavelengths = np.array([0.58-.08, 0.58+.08])*1.0e-6
        # guess at the full system optical transmission (excluding obscuration)
        optics_transmission = 0.5*np.ones(opt_trans_wavelengths.shape[0])

        # Relative linear telescope obscuration
        eta = 0.4  # guess

        # detector width is assumed to be equal to the pitch
        w_x = p
        w_y = p
        # integration time (s) - this is a maximum, the actual integration time will be
        # determined by the well fill percentage
        int_time = 30.0e-3

        # dark current density of 1 nA/cm2 guess, guess mid range for a silicon camera
        dark_current = dark_current_from_density(1e-5, w_x, w_y)

        # rms read noise (rms electrons)
        read_noise = 25.0

        # maximum ADC level (electrons)
        max_n = 96000.0

        # bit depth
        bit_depth = 11.9

        # maximum allowable well fill (see the paper for the logic behind this)
        max_well_fill = .6

        # jitter (radians) - The Olson paper says that its "good" so we'll guess 1/4 ifov rms
        s_x = 0.25*p/f
        s_y = s_x

        # drift (radians/s) - again, we'll guess that it's really good
        da_x = 100e-6
        da_y = da_x

        # etector quantum efficiency as a function of wavelength (microns)
        # for a generic high quality back-illuminated silicon array
        # https://www.photometrics.com/resources/learningzone/quantumefficiency.php
        qe_wavelengths = np.array([.3, .4, .5, .6, .7, .8, .9, 1.0, 1.1])*1.0e-6
        qe = np.array([0.05, 0.6, 0.75, 0.85, .85, .75, .5, .2, 0])

        sensor = PybsmSensor(name, D, f, p, opt_trans_wavelengths,
                             optics_transmission, eta, w_x, w_y,
                             int_time, dark_current, read_noise,
                             max_n, bit_depth, max_well_fill, s_x, s_y,
                             da_x, da_y, qe_wavelengths, qe)

        altitude = 9000.0
        # range to target
        ground_range = 60000.0

        scenario_name = 'niceda_y'
        # weather model
        ihaze = 1
        scenario = PybsmScenario(scenario_name, ihaze, altitude, ground_range)
        scenario.aircraft_speed = 100.0

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
        inst = PybsmPerturber(sensor=sensor, scenario=scenario, ground_range=10000)
        pybsm_perturber_assertions(perturb=inst.perturb, image=image, expected=expected,
                                   additional_params={'img_gsd': img_gsd})

        # Test callable
        pybsm_perturber_assertions(
            perturb=PybsmPerturber(sensor=sensor, scenario=scenario, ground_range=10000),
            image=image,
            expected=expected,
            additional_params={'img_gsd': img_gsd}
        )

    @pytest.mark.parametrize("param_name, param_value", [
        ('ground_range', 10000),
        ('ground_range', 20000),
        ('ground_range', 30000),
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
            assert i.sensor.p_x == sensor.p_x
            assert np.array_equal(i.sensor.opt_trans_wavelengths, sensor.opt_trans_wavelengths)
            assert np.array_equal(i.sensor.optics_transmission, sensor.optics_transmission)
            assert i.sensor.eta == sensor.eta
            assert i.sensor.w_x == sensor.w_x
            assert i.sensor.w_y == sensor.w_y
            assert i.sensor.int_time == sensor.int_time
            assert i.sensor.dark_current == sensor.dark_current
            assert i.sensor.read_noise == sensor.read_noise
            assert i.sensor.max_n == sensor.max_n
            assert i.sensor.bit_depth == sensor.bit_depth
            assert i.sensor.max_well_fill == sensor.max_well_fill
            assert i.sensor.s_x == sensor.s_x
            assert i.sensor.s_y == sensor.s_y
            assert i.sensor.da_x == sensor.da_x
            assert i.sensor.da_y == sensor.da_y
            assert np.array_equal(i.sensor.qe_wavelengths, sensor.qe_wavelengths)
            assert np.array_equal(i.sensor.qe, sensor.qe)

            assert i.scenario.name == scenario.name
            assert i.scenario.ihaze == scenario.ihaze
            assert i.scenario.altitude == scenario.altitude
            assert i.scenario.ground_range == scenario.ground_range
            assert i.scenario.aircraft_speed == scenario.aircraft_speed
            assert i.scenario.target_reflectance == scenario.target_reflectance
            assert i.scenario.target_temperature == scenario.target_temperature
            assert i.scenario.background_reflectance == scenario.background_reflectance
            assert i.scenario.background_temperature == scenario.background_temperature
            assert i.scenario.ha_wind_speed == scenario.ha_wind_speed
            assert i.scenario.cn2_at_1m == scenario.cn2_at_1m

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
