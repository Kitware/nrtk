import numpy as np
import pytest
import pybsm
from contextlib import nullcontext as does_not_raise
from typing import Any, ContextManager, Tuple, Sequence

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
        darkCurrent = pybsm.darkCurrentFromDensity(1e-5, wx, wy)

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

    @pytest.mark.parametrize("theta_keys, thetas, expected", [
        (["altitude"], [[1000, 2000, 3000, 4000]], ((1000,), (2000,), (3000,), (4000,))),
        (["altitude", "groundRange"], [[1000, 2000], [10000, 20000]],
            ((1000, 10000), (1000, 20000), (2000, 10000), (2000, 20000)))
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
                assert p.get_config()[theta_keys[count]] == expected[idx][count]

    @pytest.mark.parametrize("theta_keys, thetas, idx, expected_val, expectation", [
        (["altitude", "groundRange"], [[1000, 2000], [10000, 20000]], 0, (1000, 10000), does_not_raise()),
        (["altitude", "groundRange"], [[1000, 2000], [10000, 20000]], 3, (2000, 20000), does_not_raise()),
        (["altitude", "groundRange"], [[1000, 2000], [10000, 20000]], 4, (-1, -1), pytest.raises(AssertionError)),
        (["altitude", "groundRange"], [[1000, 2000], [10000, 20000]], -1, (2000, 20000), does_not_raise())
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
                assert factory[idx].get_config()[theta_keys[count]] == expected_val[count]

    @pytest.mark.parametrize("theta_keys, thetas, expected_sets", [
        (["altitude"], [[1000, 2000, 3000, 4000]], [[0], [1], [2], [3]]),
        (["altitude", "groundRange"], [[1000, 2000], [10000, 20000]], [[0, 0], [0, 1], [1, 0], [1, 1]])
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
        inst_config = inst.get_config()
        assert inst_config['theta_keys'] == theta_keys
        assert inst_config['thetas'] == thetas
        assert inst_config['sensor'] == sensor.get_config()
        assert inst_config['scenario'] == scenario.get_config()
        assert inst_config['sets'] == expected_sets
