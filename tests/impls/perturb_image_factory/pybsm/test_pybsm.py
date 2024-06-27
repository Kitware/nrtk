import numpy as np
import pytest
from pybsm.otf import dark_current_from_density
from contextlib import nullcontext as does_not_raise
from typing import Any, ContextManager, Tuple, Sequence

from smqtk_core.configuration import configuration_test_helper

from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image_factory.pybsm import CustomPybsmPerturbImageFactory


class TestStepPerturbImageFactory:
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

    @pytest.mark.parametrize("theta_keys, thetas, expected", [
        (["altitude"], [[1000, 2000, 3000, 4000]], ((1000,), (2000,), (3000,), (4000,))),
        (["altitude", "D"], [[1000, 2000], [0.5, 0.75]],
            ((1000, 0.5), (1000, 0.75), (2000, 0.5), (2000, 0.75)))
    ])
    def test_iteration(
        self,
        theta_keys: Sequence[str],
        thetas: Sequence[Any],
        expected: Tuple[Tuple[int, ...]]
    ) -> None:
        """
        Ensure factory can be iterated upon and the varied parameter matches expectations.
        """
        sensor, scenario = self.createSampleSensorandScenario()
        factory = CustomPybsmPerturbImageFactory(
            sensor=sensor,
            scenario=scenario,
            theta_keys=theta_keys,
            thetas=thetas
        )
        assert len(expected) == len(factory)
        for idx, p in enumerate(factory):
            for count, _ in enumerate(theta_keys):
                perturb_cfg = p.get_config()
                sensor_cfg = perturb_cfg["sensor"]["nrtk.impls.perturb_image.pybsm.sensor.PybsmSensor"]
                sce_cfg = perturb_cfg["scenario"]["nrtk.impls.perturb_image.pybsm.scenario.PybsmScenario"]

                if theta_keys[count] in sensor_cfg:
                    assert sensor_cfg[theta_keys[count]] == expected[idx][count]
                elif theta_keys[count] in sce_cfg:
                    assert sce_cfg[theta_keys[count]] == expected[idx][count]
                # elif theta_keys[count] in perturb_cfg:  # reflectance_range
                #    assert perturb_cfg[theta_keys[count]] == expected[idx][count]
                else:  # pragma: no cover
                    assert False, "Parameter not found in config"

    @pytest.mark.parametrize("theta_keys, thetas, idx, expected_val, expectation", [
        (["altitude", "D"], [[1000, 2000], [0.5, 0.75]], 0, (1000, 0.5), does_not_raise()),
        (["altitude", "D"], [[1000, 2000], [10000, 20000]], 3, (2000, 20000), does_not_raise()),
        (["altitude", "D"], [[1000, 2000], [10000, 20000]], 4, (-1, -1), pytest.raises(IndexError)),
        (["altitude", "D"], [[1000, 2000], [10000, 20000]], -1, (2000, 20000), does_not_raise())
    ], ids=["first idx", "last idx", "idx == len", "neg idx"])
    def test_indexing(
        self,
        theta_keys: Sequence[str],
        thetas: Sequence[Any],
        idx: int,
        expected_val: Tuple[int, ...],
        expectation: ContextManager
    ) -> None:
        """
        Ensure it is possible to access a perturber instance via indexing.
        """
        sensor, scenario = self.createSampleSensorandScenario()
        factory = CustomPybsmPerturbImageFactory(
            sensor=sensor,
            scenario=scenario,
            theta_keys=theta_keys,
            thetas=thetas
        )
        with expectation:
            for count, _ in enumerate(theta_keys):
                perturb_cfg = factory[idx].get_config()
                sensor_cfg = perturb_cfg["sensor"]["nrtk.impls.perturb_image.pybsm.sensor.PybsmSensor"]
                sce_cfg = perturb_cfg["scenario"]["nrtk.impls.perturb_image.pybsm.scenario.PybsmScenario"]

                if theta_keys[count] in sensor_cfg:
                    assert sensor_cfg[theta_keys[count]] == expected_val[count]
                elif theta_keys[count] in sce_cfg:
                    assert sce_cfg[theta_keys[count]] == expected_val[count]
                # elif theta_keys[count] in perturb_cfg:  # reflectance_range
                #     assert perturb_cfg[theta_keys[count]] == expected_val[count]
                else:  # pragma: no cover
                    assert False, "Parameter not found in config"

    @pytest.mark.parametrize("theta_keys, thetas, expected_sets", [
        (["altitude"], [[1000, 2000, 3000, 4000]], [[0], [1], [2], [3]]),
        (["altitude", "ground_range"], [[1000, 2000], [10000, 20000]], [[0, 0], [0, 1], [1, 0], [1, 1]])
    ])
    def test_configuration(
        self,
        theta_keys: Sequence[str],
        thetas: Sequence[Any],
        expected_sets: Sequence[Sequence[int]]
    ) -> None:
        """
        Test configuration stability.
        """
        sensor, scenario = self.createSampleSensorandScenario()
        inst = CustomPybsmPerturbImageFactory(
            sensor=sensor,
            scenario=scenario,
            theta_keys=theta_keys,
            thetas=thetas
        )

        for i in configuration_test_helper(inst):
            assert i.theta_keys == theta_keys
            assert i.thetas == thetas

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

            assert i.sets == expected_sets
