import json
from collections.abc import Sequence
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from smqtk_core.configuration import (
    configuration_test_helper,
    from_config_dict,
    to_config_dict,
)

from nrtk.impls.perturb_image.optical.pybsm_perturber import PybsmPerturber
from nrtk.impls.perturb_image_factory.perturber_multivariate_factory import PerturberMultivariateFactory
from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from tests.impls.test_pybsm_utils import create_sample_sensor_and_scenario
from tests.test_utils import DummyPerturber


class TestPerturberMultivariateFactory:
    @pytest.mark.parametrize(
        ("perturber", "theta_keys", "thetas", "expected"),
        [
            (
                DummyPerturber,
                ["param1"],
                [[1, 3, 5]],
                ((1,), (3,), (5,)),
            ),
            (
                DummyPerturber,
                ["param1", "param2"],
                [[1, 3], [2, 4]],
                ((1, 2), (1, 4), (3, 2), (3, 4)),
            ),
        ],
    )
    def test_iteration(
        self,
        perturber: type[PerturbImage],
        theta_keys: Sequence[str],
        thetas: Sequence[Any],
        expected: tuple[tuple[int, ...]],
    ) -> None:
        """Ensure factory can be iterated upon and the varied parameter matches expectations."""
        factory = PerturberMultivariateFactory(perturber=perturber, theta_keys=theta_keys, thetas=thetas)
        assert len(expected) == len(factory)
        for idx, p in enumerate(factory):
            for count, _ in enumerate(theta_keys):
                perturb_cfg = p.get_config()
                assert perturb_cfg[theta_keys[count]] == expected[idx][count]

    @pytest.mark.parametrize(
        ("perturber", "theta_keys", "thetas", "idx", "expected_val", "expectation"),
        [
            (
                DummyPerturber,
                ["param1", "param2"],
                [[1, 3], [2, 4]],
                0,
                (1, 2),
                does_not_raise(),
            ),
            (
                DummyPerturber,
                ["param1", "param2"],
                [[1, 3], [2, 4]],
                3,
                (3, 4),
                does_not_raise(),
            ),
            (
                DummyPerturber,
                ["param1", "param2"],
                [[1, 3], [2, 4]],
                4,
                (-1, -1),
                pytest.raises(IndexError),
            ),
            (
                DummyPerturber,
                ["param1", "param2"],
                [[1, 3], [2, 4]],
                -1,
                (3, 4),
                does_not_raise(),
            ),
        ],
        ids=["first idx", "last idx", "idx == len", "neg idx"],
    )
    def test_indexing(
        self,
        perturber: type[PerturbImage],
        theta_keys: Sequence[str],
        thetas: Sequence[Any],
        idx: int,
        expected_val: tuple[int, ...],
        expectation: AbstractContextManager,
    ) -> None:
        """Ensure it is possible to access a perturber instance via indexing."""
        factory = PerturberMultivariateFactory(perturber=perturber, theta_keys=theta_keys, thetas=thetas)
        with expectation:
            for count, _ in enumerate(theta_keys):
                perturb_cfg = factory[idx].get_config()
                assert perturb_cfg[theta_keys[count]] == expected_val[count]

    @pytest.mark.parametrize(
        ("perturber", "theta_keys", "thetas", "expected_sets"),
        [
            (DummyPerturber, ["param1"], [[1, 2, 3, 4]], [[0], [1], [2], [3]]),
            (
                DummyPerturber,
                ["param1", "param2"],
                [[1, 3], [2, 4]],
                [[0, 0], [0, 1], [1, 0], [1, 1]],
            ),
        ],
    )
    def test_configuration(
        self,
        perturber: type[PerturbImage],
        theta_keys: Sequence[str],
        thetas: Sequence[Any],
        expected_sets: Sequence[Sequence[int]],
    ) -> None:
        """Test configuration stability."""
        inst = PerturberMultivariateFactory(perturber=perturber, theta_keys=theta_keys, thetas=thetas)

        for i in configuration_test_helper(inst):
            assert i.theta_keys == theta_keys
            assert i.thetas == thetas
            assert i.sets == expected_sets

    @pytest.mark.parametrize(
        ("perturber", "theta_keys", "thetas"),
        [
            (
                DummyPerturber,
                ["param1"],
                [[1, 2, 3, 4]],
            ),
            (
                DummyPerturber,
                ["param1", "param2"],
                [[1, 3], [2, 4]],
            ),
        ],
    )
    def test_hydration(
        self,
        perturber: type[PerturbImage],
        tmp_path: Path,
        theta_keys: Sequence[str],
        thetas: Sequence[Any],
    ) -> None:
        """Test configuration hydration using from_config_dict."""
        original_factory = PerturberMultivariateFactory(perturber=perturber, theta_keys=theta_keys, thetas=thetas)

        original_factory_config = original_factory.get_config()

        config_file_path = tmp_path / "config.json"
        with open(str(config_file_path), "w") as f:
            json.dump(to_config_dict(original_factory), f)

        with open(str(config_file_path)) as config_file:
            config = json.load(config_file)
            hydrated_factory = from_config_dict(config=config, type_iter=PerturbImageFactory.get_impls())
            hydrated_factory_config = hydrated_factory.get_config()

            assert original_factory_config == hydrated_factory_config


@pytest.mark.skipif(
    not PybsmPerturber.is_usable(),
    reason="not PybsmPerturber.is_usable()",
)
class TestPerturberStepFactory:
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
        expected: tuple[tuple[int, ...]],
    ) -> None:
        """Ensure factory can be iterated upon and the varied parameter matches expectations."""
        sensor_and_scenario = create_sample_sensor_and_scenario()

        factory = PerturberMultivariateFactory(
            perturber=PybsmPerturber,
            theta_keys=theta_keys,
            thetas=thetas,
            perturber_kwargs=sensor_and_scenario,
        )
        assert len(expected) == len(factory)
        for idx, p in enumerate(factory):
            for count, theta_key in enumerate(theta_keys):
                perturb_cfg = p.get_config()
                if theta_key in perturb_cfg:
                    assert perturb_cfg[theta_key] == expected[idx][count]

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
        expected_val: tuple[int, ...],
        expectation: AbstractContextManager,
    ) -> None:
        """Ensure it is possible to access a perturber instance via indexing."""
        sensor_and_scenario = create_sample_sensor_and_scenario()

        factory = PerturberMultivariateFactory(
            perturber=PybsmPerturber,
            theta_keys=theta_keys,
            thetas=thetas,
            perturber_kwargs=sensor_and_scenario,
        )
        with expectation:
            for count, theta_key in enumerate(theta_keys):
                perturb_cfg = factory[idx].get_config()
                if theta_key in perturb_cfg:
                    assert perturb_cfg[theta_key] == expected_val[count]

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
        sensor_and_scenario = create_sample_sensor_and_scenario()

        inst = PerturberMultivariateFactory(
            perturber=PybsmPerturber,
            theta_keys=theta_keys,
            thetas=thetas,
            perturber_kwargs=sensor_and_scenario,
        )

        for i in configuration_test_helper(inst):
            assert i.theta_keys == theta_keys
            assert i.thetas == thetas

            for param_name, param_value in i.perturber_kwargs.items():
                if isinstance(param_value, np.ndarray):
                    assert np.allclose(sensor_and_scenario[param_name], param_value)
                else:
                    assert sensor_and_scenario[param_name] == param_value

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
        original_factory = PerturberMultivariateFactory(
            perturber=PybsmPerturber,
            theta_keys=theta_keys,
            thetas=thetas,
        )

        original_factory_config = original_factory.get_config()

        config_file_path = tmp_path / "config.json"
        with open(str(config_file_path), "w") as f:
            json.dump(to_config_dict(original_factory), f)

        with open(str(config_file_path)) as config_file:
            config = json.load(config_file)
            hydrated_factory = from_config_dict(config=config, type_iter=PerturbImageFactory.get_impls())
            hydrated_factory_config = hydrated_factory.get_config()

            assert original_factory_config == hydrated_factory_config
