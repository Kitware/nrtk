from contextlib import nullcontext as does_not_raise
from typing import Any, ContextManager, Dict, Tuple

import numpy as np
import pytest
from PIL import Image
from pybsm.otf import dark_current_from_density
from smqtk_core.configuration import configuration_test_helper

from nrtk.impls.perturb_image.pybsm.jitter_otf_perturber import JitterOTFPerturber
from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor

from ..test_perturber_utils import pybsm_perturber_assertions

INPUT_IMG_FILE = (
    "./examples/pybsm/data/M-41 Walker Bulldog (USA) width 319cm height 272cm.tiff"
)
EXPECTED_DEFAULT_IMG_FILE = (
    "./tests/impls/perturb_image/pybsm/data/jitter_otf_default_expected_output.tiff"
)
EXPECTED_PROVIDED_IMG_FILE = (
    "./tests/impls/perturb_image/pybsm/data/jitter_otf_provided_expected_output.tiff"
)


class TestOTFPerturber:
    def create_sample_sensor_and_scenario(self) -> Tuple[PybsmSensor, PybsmScenario]:
        name = "L32511x"

        # telescope focal length (m)
        f = 4
        # Telescope diameter (m)
        D = 275e-3  # noqa:N806

        # detector pitch (m)
        p = 0.008e-3

        # Optical system transmission, red  band first (m)
        opt_trans_wavelengths = np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6
        # guess at the full system optical transmission (excluding obscuration)
        optics_transmission = 0.5 * np.ones(opt_trans_wavelengths.shape[0])

        # Relative linear telescope obscuration
        eta = 0.4  # guess

        # detector width is assumed to be equal to the pitch
        w_x = p
        w_y = p
        # integration time (s) - this is a maximum, the actual integration
        # time will be determined by the well fill percentage
        int_time = 30.0e-3

        # dark current density of 1 nA/cm2 guess, guess mid range for a
        # silicon camera
        dark_current = dark_current_from_density(1e-5, w_x, w_y)

        # rms read noise (rms electrons)
        read_noise = 25.0

        # maximum ADC level (electrons)
        max_n = 96000.0

        # bit depth
        bit_depth = 11.9

        # maximum allowable well fill (see the paper for the logic behind this)
        max_well_fill = 0.6

        # jitter (radians) - The Olson paper says that its "good"
        # so we'll guess 1/4 ifov rms
        s_x = 0.25 * p / f
        s_y = s_x

        # drift (radians/s) - again, we'll guess that it's really good
        da_x = 100e-6
        da_y = da_x

        # etector quantum efficiency as a function of wavelength (microns)
        # for a generic high quality back-illuminated silicon array
        # https://www.photometrics.com/resources/learningzone/quantumefficiency.php
        qe_wavelengths = (
            np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]) * 1.0e-6
        )
        qe = np.array([0.05, 0.6, 0.75, 0.85, 0.85, 0.75, 0.5, 0.2, 0])

        sensor = PybsmSensor(
            name,
            D,
            f,
            p,
            opt_trans_wavelengths,
            optics_transmission,
            eta,
            w_x,
            w_y,
            int_time,
            dark_current,
            read_noise,
            max_n,
            bit_depth,
            max_well_fill,
            s_x,
            s_y,
            da_x,
            da_y,
            qe_wavelengths,
            qe,
        )

        altitude = 9000.0
        # range to target
        ground_range = 60000.0

        scenario_name = "niceday"
        # weather model
        ihaze = 1
        scenario = PybsmScenario(scenario_name, ihaze, altitude, ground_range)
        scenario.aircraft_speed = 100.0

        return sensor, scenario

    def test_provided_consistency(self) -> None:
        """Run on a dummy image to ensure output matches precomputed results."""
        image = np.array(Image.open(INPUT_IMG_FILE))
        expected = np.array(Image.open(EXPECTED_PROVIDED_IMG_FILE))
        img_gsd = 3.19 / 160.0
        sensor, scenario = self.create_sample_sensor_and_scenario()
        # Test perturb interface directly
        inst = JitterOTFPerturber(sensor=sensor, scenario=scenario)
        pybsm_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=expected,
            additional_params={"img_gsd": img_gsd},
        )

        # Test callable
        pybsm_perturber_assertions(
            perturb=JitterOTFPerturber(sensor=sensor, scenario=scenario),
            image=image,
            expected=expected,
            additional_params={"img_gsd": img_gsd},
        )

    def test_default_consistency(self) -> None:
        """Run on a dummy image to ensure output matches precomputed results."""
        image = np.array(Image.open(INPUT_IMG_FILE))
        expected = np.array(Image.open(EXPECTED_DEFAULT_IMG_FILE))
        # Test perturb interface directly
        inst = JitterOTFPerturber()
        pybsm_perturber_assertions(perturb=inst.perturb, image=image, expected=expected)

        # Test callable
        pybsm_perturber_assertions(
            perturb=JitterOTFPerturber(), image=image, expected=expected
        )

    @pytest.mark.parametrize("s_x", [0.5, 1.5])
    @pytest.mark.parametrize("s_y", [0.5, 1.5])
    def test_provided_reproducibility(self, s_x: float, s_y: float) -> None:
        """Ensure results are reproducible."""
        # Test perturb interface directly
        image = np.array(Image.open(INPUT_IMG_FILE))
        sensor, scenario = self.create_sample_sensor_and_scenario()
        inst = JitterOTFPerturber(sensor=sensor, scenario=scenario, s_x=s_x, s_y=s_y)
        img_gsd = 3.19 / 160.0
        out_image = pybsm_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=None,
            additional_params={"img_gsd": img_gsd},
        )
        pybsm_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=out_image,
            additional_params={"img_gsd": img_gsd},
        )

    def test_default_reproducibility(self) -> None:
        """Ensure results are reproducible."""
        # Test perturb interface directly
        image = np.array(Image.open(INPUT_IMG_FILE))
        inst = JitterOTFPerturber()
        out_image = pybsm_perturber_assertions(
            perturb=inst.perturb, image=image, expected=None
        )
        pybsm_perturber_assertions(
            perturb=inst.perturb, image=image, expected=out_image
        )

    @pytest.mark.parametrize(
        ("additional_params", "expectation"),
        [
            ({"img_gsd": 3.19 / 160.0}, does_not_raise()),
            (
                {},
                pytest.raises(
                    ValueError, match=r"'img_gsd' must be present in image metadata"
                ),
            ),
        ],
    )
    def test_provided_additional_params(
        self, additional_params: Dict[str, Any], expectation: ContextManager
    ) -> None:
        """Test variations of additional params."""
        sensor, scenario = self.create_sample_sensor_and_scenario()
        perturber = JitterOTFPerturber(sensor=sensor, scenario=scenario)
        image = np.array(Image.open(INPUT_IMG_FILE))
        with expectation:
            _ = perturber(image, additional_params)

    @pytest.mark.parametrize(
        ("additional_params", "expectation"),
        [
            ({}, does_not_raise()),
        ],
    )
    def test_default_additional_params(
        self, additional_params: Dict[str, Any], expectation: ContextManager
    ) -> None:
        """Test variations of additional params."""
        perturber = JitterOTFPerturber()
        image = np.array(Image.open(INPUT_IMG_FILE))
        with expectation:
            _ = perturber(image, additional_params)

    @pytest.mark.parametrize("s_x", [0.5, 1.5])
    @pytest.mark.parametrize("s_y", [0.5, 1.5])
    def test_provided_sx_sy_reproducibility(
        self,
        s_x: float,
        s_y: float,
    ) -> None:
        """Ensure results are reproducible."""
        # Test perturb interface directly
        image = np.array(Image.open(INPUT_IMG_FILE))
        sensor, scenario = self.create_sample_sensor_and_scenario()
        inst = JitterOTFPerturber(sensor=sensor, scenario=scenario, s_x=s_x, s_y=s_y)
        img_gsd = 3.19 / 160.0
        out_image = pybsm_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=None,
            additional_params={"img_gsd": img_gsd},
        )
        pybsm_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=out_image,
            additional_params={"img_gsd": img_gsd},
        )

    @pytest.mark.parametrize("s_x", [0.5])
    @pytest.mark.parametrize("s_y", [0.5])
    def test_sx_sy_configuration(
        self,
        s_x: float,
        s_y: float,
    ) -> None:
        """Test configuration stability."""
        inst = JitterOTFPerturber(s_x=s_x, s_y=s_y)
        for i in configuration_test_helper(inst):
            assert i.s_x == s_x
            assert i.s_y == s_y

    def test_sensor_scenario_configuration(self) -> None:
        """Test configuration stability."""
        sensor, scenario = self.create_sample_sensor_and_scenario()
        inst = JitterOTFPerturber(sensor=sensor, scenario=scenario)
        for i in configuration_test_helper(inst):
            if i.sensor:
                assert i.sensor.name == sensor.name
                assert i.sensor.D == sensor.D
                assert i.sensor.f == sensor.f
                assert i.sensor.p_x == sensor.p_x
                assert np.array_equal(
                    i.sensor.opt_trans_wavelengths, sensor.opt_trans_wavelengths
                )
                assert np.array_equal(
                    i.sensor.optics_transmission, sensor.optics_transmission
                )
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
            if i.scenario:
                assert i.scenario.name == scenario.name
                assert i.scenario.ihaze == scenario.ihaze
                assert i.scenario.altitude == scenario.altitude
                assert i.scenario.ground_range == scenario.ground_range
                assert i.scenario.aircraft_speed == scenario.aircraft_speed
                assert i.scenario.target_reflectance == scenario.target_reflectance
                assert i.scenario.target_temperature == scenario.target_temperature
                assert (
                    i.scenario.background_reflectance == scenario.background_reflectance
                )
                assert (
                    i.scenario.background_temperature == scenario.background_temperature
                )
                assert i.scenario.ha_wind_speed == scenario.ha_wind_speed
                assert i.scenario.cn2_at_1m == scenario.cn2_at_1m

    @pytest.mark.parametrize("s_x", [0.5])
    @pytest.mark.parametrize("s_y", [0.5])
    def test_overall_configuration(
        self,
        s_x: float,
        s_y: float,
    ) -> None:
        """Test configuration stability."""
        sensor, scenario = self.create_sample_sensor_and_scenario()
        inst = JitterOTFPerturber(sensor=sensor, scenario=scenario, s_x=s_x, s_y=s_y)
        for i in configuration_test_helper(inst):
            assert i.s_x == s_x
            assert i.s_y == s_y
            if i.sensor:
                assert i.sensor.name == sensor.name
                assert i.sensor.D == sensor.D
                assert i.sensor.f == sensor.f
                assert i.sensor.p_x == sensor.p_x
                assert np.array_equal(
                    i.sensor.opt_trans_wavelengths, sensor.opt_trans_wavelengths
                )
                assert np.array_equal(
                    i.sensor.optics_transmission, sensor.optics_transmission
                )
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
            if i.scenario:
                assert i.scenario.name == scenario.name
                assert i.scenario.ihaze == scenario.ihaze
                assert i.scenario.altitude == scenario.altitude
                assert i.scenario.ground_range == scenario.ground_range
                assert i.scenario.aircraft_speed == scenario.aircraft_speed
                assert i.scenario.target_reflectance == scenario.target_reflectance
                assert i.scenario.target_temperature == scenario.target_temperature
                assert (
                    i.scenario.background_reflectance == scenario.background_reflectance
                )
                assert (
                    i.scenario.background_temperature == scenario.background_temperature
                )
                assert i.scenario.ha_wind_speed == scenario.ha_wind_speed
                assert i.scenario.cn2_at_1m == scenario.cn2_at_1m
