from typing import Tuple

import numpy as np
from pybsm.otf import dark_current_from_density

from nrtk.impls.image_metric.niirs_image_metric import NIIRSImageMetric
from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor


class TestSNRImageMetric:
    """This class contains the unit tests for the functionality of the NIIRSImageMetric impl."""

    def create_sample_sensor_and_scenario(self) -> Tuple[PybsmSensor, PybsmScenario]:

        name = "L32511x"

        # telescope focal length (m)
        f = 4
        # Telescope diameter (m)
        D = 275e-3  # noqa: N806

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
        max_well_fill = 0.6

        # jitter (radians) - The Olson paper says that its "good" so we'll guess 1/4 ifov rms
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

    def test_consistency(self) -> None:
        expected_niirs = 5.5974654005905835
        sensor, scenario = self.create_sample_sensor_and_scenario()
        niirs_metric = NIIRSImageMetric(sensor=sensor, scenario=scenario)
        # Test metric interface directly
        niirs = niirs_metric.compute()
        assert expected_niirs == niirs

        # Test callable
        assert expected_niirs == niirs_metric()

    def test_reproducibility(self) -> None:
        """Ensure results are reproducible."""
        # Test metric interface directly
        sensor, scenario = self.create_sample_sensor_and_scenario()

        niirs_ouput = NIIRSImageMetric(sensor=sensor, scenario=scenario).compute()
        niirs_metric = NIIRSImageMetric(sensor=sensor, scenario=scenario)

        assert niirs_ouput == niirs_metric.compute()

    def test_get_config(self) -> None:
        """Ensure get_config is correct."""
        sensor, scenario = self.create_sample_sensor_and_scenario()
        niirs_metric = NIIRSImageMetric(sensor=sensor, scenario=scenario)

        niirs_metric_config = niirs_metric.get_config()
        assert niirs_metric_config["sensor"] == {
            "nrtk.impls.perturb_image.pybsm.sensor.PybsmSensor": sensor.get_config(),
            "type": "nrtk.impls.perturb_image.pybsm.sensor.PybsmSensor",
        }
        assert niirs_metric_config["scenario"] == {
            "nrtk.impls.perturb_image.pybsm.scenario.PybsmScenario": scenario.get_config(),
            "type": "nrtk.impls.perturb_image.pybsm.scenario.PybsmScenario",
        }
