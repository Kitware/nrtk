from __future__ import annotations

import json
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Any

import pytest
from smqtk_core.configuration import (
    configuration_test_helper,
    from_config_dict,
    to_config_dict,
)

from nrtk.impls.perturb_image.photometric.noise import SaltNoisePerturber
from nrtk.impls.perturb_image_factory.generic.linspace import (
    LinspacePerturbImageFactory,
)
from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from tests.impls import DATA_DIR
from tests.test_utils import DummyPerturber


class TestLinspacePerturbImageFactory:
    @pytest.mark.parametrize(
        ("perturber", "theta_key", "start", "stop", "num", "expected"),
        [
            (DummyPerturber, "param1", 1, 3, 5, (1, 1.5, 2.0, 2.5, 3)),
            (DummyPerturber, "param2", 3, 9, 2, (3, 9)),
            (DummyPerturber, "param1", 4, 4, 1, (4,)),
        ],
    )
    def test_iteration(
        self,
        perturber: type[PerturbImage],
        theta_key: str,
        start: int,
        stop: int,
        num: int,
        expected: tuple[int, ...],
    ) -> None:
        """Ensure factory can be iterated upon and the varied parameter matches expectations."""
        factory = LinspacePerturbImageFactory(
            perturber=perturber,
            theta_key=theta_key,
            start=start,
            stop=stop,
            num=num,
        )
        assert len(expected) == len(factory)
        for idx, p in enumerate(factory):
            print(p.get_config(), expected)
            assert p.get_config()[theta_key] == expected[idx]

    @pytest.mark.parametrize(
        (
            "perturber",
            "theta_key",
            "start",
            "stop",
            "num",
            "idx",
            "expected_val",
            "expectation",
        ),
        [
            (DummyPerturber, "param1", 1, 6, 10, 0, 1, does_not_raise()),
            (DummyPerturber, "param1", 1, 6, 10, 3, 2.666666666666667, does_not_raise()),
            (DummyPerturber, "param1", 1, 6, 2, 3, -1, pytest.raises(IndexError)),
            (DummyPerturber, "param1", 1, 6, 2, -1, -1, pytest.raises(IndexError)),
            (DummyPerturber, "param1", 4, 3, 1, 0, 4, does_not_raise()),
        ],
        ids=["first idx", "last idx", "idx == len", "neg idx", "empty iter"],
    )
    def test_indexing(
        self,
        perturber: type[PerturbImage],
        theta_key: str,
        start: int,
        stop: int,
        num: int,
        idx: int,
        expected_val: int,
        expectation: AbstractContextManager,
    ) -> None:
        """Ensure it is possible to access a perturber instance via indexing."""
        factory = LinspacePerturbImageFactory(
            perturber=perturber,
            theta_key=theta_key,
            start=start,
            stop=stop,
            num=num,
        )
        with expectation:
            print(idx, factory[idx].get_config(), expected_val)
            assert factory[idx].get_config()[theta_key] == expected_val

    @pytest.mark.parametrize(
        ("start", "stop", "num", "endpoint", "expected"),
        [
            (0.0, 1.0, 2, True, [0.0, 1.0]),
            (0.0, 1.0, 2, False, [0.0, 0.5]),
            (2.0, 1.0, 1, False, [2.0]),
            (1.0, 1.0, 3, False, [1.0, 1.0, 1.0]),
        ],
    )
    def test_thetas(
        self,
        start: float,
        stop: float,
        num: int,
        endpoint: bool,
        expected: list[float],
    ) -> None:
        """Test the generated theta values."""
        factory = LinspacePerturbImageFactory(
            perturber=DummyPerturber,
            theta_key="param1",
            start=start,
            stop=stop,
            num=num,
            endpoint=endpoint,
        )
        assert factory.thetas == expected

    @pytest.mark.parametrize(
        ("perturber", "theta_key", "start", "stop", "num"),
        [
            (DummyPerturber, "param1", 1.0, 5.0, 2),
            (DummyPerturber, "param2", 3.0, 9.0, 3),
        ],
    )
    def test_configuration(
        self,
        perturber: type[PerturbImage],
        theta_key: str,
        start: float,
        stop: float,
        num: int,
        endpoint: bool = True,
    ) -> None:
        """Test configuration stability."""
        inst = LinspacePerturbImageFactory(perturber=perturber, theta_key=theta_key, start=start, stop=stop, num=num)
        for i in configuration_test_helper(inst):
            assert i.perturber == perturber
            assert i.theta_key == theta_key
            assert i.start == start
            assert i.stop == stop
            assert i.num == num
            assert start in i.thetas
            assert stop not in i.thetas if not endpoint else stop in i.thetas

    @pytest.mark.parametrize(
        ("kwargs", "expectation"),
        [
            (
                {
                    "perturber": DummyPerturber,
                    "theta_key": "param1",
                    "start": 1,
                    "stop": 2,
                },
                does_not_raise(),
            ),
            (
                {
                    "perturber": DummyPerturber(param1=1, param2=2),
                    "theta_key": "param2",
                    "start": 1,
                    "stop": 2,
                },
                pytest.raises(TypeError, match=r"Passed a perturber instance, expected type"),
            ),
        ],
    )
    def test_configuration_bounds(self, kwargs: dict[str, Any], expectation: AbstractContextManager) -> None:
        """Test that an exception is properly raised (or not) based on argument value."""
        with expectation:
            LinspacePerturbImageFactory(**kwargs)

    @pytest.mark.parametrize(
        ("perturber", "theta_key", "start", "stop", "num"),
        [
            (DummyPerturber, "param1", 1.0, 5.0, 2),
            (DummyPerturber, "param2", 3.0, 9.0, 3),
        ],
    )
    def test_hydration(
        self,
        tmp_path: Path,
        perturber: type[PerturbImage],
        theta_key: str,
        start: float,
        stop: float,
        num: int,
    ) -> None:
        """Test configuration hydration using from_config_dict."""
        original_factory = LinspacePerturbImageFactory(
            perturber=perturber,
            theta_key=theta_key,
            start=start,
            stop=stop,
            num=num,
        )

        original_factory_config = original_factory.get_config()

        print(original_factory_config)

        config_file_path = tmp_path / "config.json"
        with open(str(config_file_path), "w") as f:
            json.dump(to_config_dict(original_factory), f)

        with open(str(config_file_path)) as config_file:
            config = json.load(config_file)
            hydrated_factory = from_config_dict(config=config, type_iter=PerturbImageFactory.get_impls())
            hydrated_factory_config = hydrated_factory.get_config()

            assert original_factory_config == hydrated_factory_config

    @pytest.mark.parametrize(
        ("config_file_name", "expectation"),
        [
            pytest.param(
                "nrtk_noise_config.json",
                does_not_raise(),
                marks=pytest.mark.skipif(not SaltNoisePerturber.is_usable(), reason="SaltNoisePerturber unusable."),
            ),
            (
                "nrtk_bad_linspace_config.json",
                pytest.raises(ValueError, match=r"is not a valid perturber."),
            ),
        ],
    )
    def test_hydration_bounds(self, config_file_name: str, expectation: AbstractContextManager) -> None:
        """Test that an exception is properly raised (or not) based on argument value."""
        with expectation, open(str(DATA_DIR / config_file_name)) as config_file:
            config = json.load(config_file)
            from_config_dict(config=config, type_iter=PerturbImageFactory.get_impls())
