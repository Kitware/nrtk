import json
import unittest.mock as mock
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Any, ContextManager, Sequence, Tuple

import numpy as np
import pytest
from smqtk_core.configuration import (
    configuration_test_helper,
    from_config_dict,
    to_config_dict,
)

from nrtk.impls.perturb_image_factory.pybsm import CustomPybsmPerturbImageFactory
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory

from ...test_pybsm_utils import create_sample_sensor_and_scenario

DATA_DIR = Path(__file__).parents[3] / "data"
NRTK_PYBSM_CONFIG = DATA_DIR / "nrtk_pybsm_config.json"


@pytest.mark.skipif(
    not CustomPybsmPerturbImageFactory.is_usable(),
    reason="OpenCV not found. Please install 'nrtk[graphics]' or `nrtk[headless]`.",
)
class TestStepPerturbImageFactory:
    @pytest.mark.parametrize(
        ("theta_keys", "thetas", "expected"),
        [
            (
                ["altitude"],
                [[1000, 2000, 3000, 4000]],
                ((1000,), (2000,), (3000,), (4000,)),
            ),
            (
                ["altitude", "D"],
                [[1000, 2000], [0.5, 0.75]],
                ((1000, 0.5), (1000, 0.75), (2000, 0.5), (2000, 0.75)),
            ),
        ],
    )
    def test_iteration(
        self,
        theta_keys: Sequence[str],
        thetas: Sequence[Any],
        expected: Tuple[Tuple[int, ...]],
    ) -> None:
        """Ensure factory can be iterated upon and the varied parameter matches expectations."""
        sensor, scenario = create_sample_sensor_and_scenario()
        factory = CustomPybsmPerturbImageFactory(sensor=sensor, scenario=scenario, theta_keys=theta_keys, thetas=thetas)
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
                    pytest.fail("Parameter not found in config")

    @pytest.mark.parametrize(
        ("theta_keys", "thetas", "idx", "expected_val", "expectation"),
        [
            (
                ["altitude", "D"],
                [[1000, 2000], [0.5, 0.75]],
                0,
                (1000, 0.5),
                does_not_raise(),
            ),
            (
                ["altitude", "D"],
                [[1000, 2000], [10000, 20000]],
                3,
                (2000, 20000),
                does_not_raise(),
            ),
            (
                ["altitude", "D"],
                [[1000, 2000], [10000, 20000]],
                4,
                (-1, -1),
                pytest.raises(IndexError),
            ),
            (
                ["altitude", "D"],
                [[1000, 2000], [10000, 20000]],
                -1,
                (2000, 20000),
                does_not_raise(),
            ),
        ],
        ids=["first idx", "last idx", "idx == len", "neg idx"],
    )
    def test_indexing(
        self,
        theta_keys: Sequence[str],
        thetas: Sequence[Any],
        idx: int,
        expected_val: Tuple[int, ...],
        expectation: ContextManager,
    ) -> None:
        """Ensure it is possible to access a perturber instance via indexing."""
        sensor, scenario = create_sample_sensor_and_scenario()
        factory = CustomPybsmPerturbImageFactory(sensor=sensor, scenario=scenario, theta_keys=theta_keys, thetas=thetas)
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
                    pytest.fail("Parameter not found in config")

    @pytest.mark.parametrize(
        ("theta_keys", "thetas", "expected_sets"),
        [
            (["altitude"], [[1000, 2000, 3000, 4000]], [[0], [1], [2], [3]]),
            (
                ["altitude", "ground_range"],
                [[1000, 2000], [10000, 20000]],
                [[0, 0], [0, 1], [1, 0], [1, 1]],
            ),
        ],
    )
    def test_configuration(
        self,
        theta_keys: Sequence[str],
        thetas: Sequence[Any],
        expected_sets: Sequence[Sequence[int]],
    ) -> None:
        """Test configuration stability."""
        sensor, scenario = create_sample_sensor_and_scenario()
        inst = CustomPybsmPerturbImageFactory(sensor=sensor, scenario=scenario, theta_keys=theta_keys, thetas=thetas)

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

    @pytest.mark.parametrize(
        ("theta_keys", "thetas"),
        [
            (
                ["altitude"],
                [[1000, 2000, 3000, 4000]],
            ),
            (
                ["altitude", "D"],
                [[1000, 2000], [0.5, 0.75]],
            ),
        ],
    )
    def test_hydration(
        self,
        tmp_path: Path,
        theta_keys: Sequence[str],
        thetas: Sequence[Any],
    ) -> None:
        """Test configuration hydration using from_config_dict."""
        sensor, scenario = create_sample_sensor_and_scenario()
        original_factory = CustomPybsmPerturbImageFactory(
            sensor=sensor, scenario=scenario, theta_keys=theta_keys, thetas=thetas
        )

        original_factory_config = original_factory.get_config()

        config_file_path = tmp_path / "config.json"
        with open(str(config_file_path), "w") as f:
            json.dump(to_config_dict(original_factory), f)

        with open(str(config_file_path)) as config_file:
            config = json.load(config_file)
            hydrated_factory = from_config_dict(config, PerturbImageFactory.get_impls())
            hydrated_factory_config = hydrated_factory.get_config()

            assert original_factory_config == hydrated_factory_config


@mock.patch.object(CustomPybsmPerturbImageFactory, "is_usable")
def test_missing_deps(mock_is_usable) -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    mock_is_usable.return_value = False
    assert not CustomPybsmPerturbImageFactory.is_usable()
    sensor, scenario = create_sample_sensor_and_scenario()
    theta_keys = ["altitude"]
    thetas = [[1000, 2000, 3000, 4000]]
    with pytest.raises(ImportError, match=r"OpenCV not found"):
        CustomPybsmPerturbImageFactory(sensor=sensor, scenario=scenario, theta_keys=theta_keys, thetas=thetas)
