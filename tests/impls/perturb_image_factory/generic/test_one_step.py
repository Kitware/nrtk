import json
from collections.abc import Hashable, Iterable
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest
from smqtk_core.configuration import (
    configuration_test_helper,
    from_config_dict,
    to_config_dict,
)
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from syrupy.assertion import SnapshotAssertion

from nrtk.impls.perturb_image_factory.generic.one_step import OneStepPerturbImageFactory
from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory

DATA_DIR = Path(__file__).parents[3] / "data"
INPUT_IMG_FILE_PATH = "./examples/pybsm/data/M-41 Walker Bulldog (USA) width 319cm height 272cm.tiff"


class DummyPerturber(PerturbImage):
    def __init__(self, param_1: int = 1, param_2: int = 2) -> None:
        self.param_1 = param_1
        self.param_2 = param_2

    def perturb(
        self,
        image: np.ndarray,
        boxes: Optional[Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]] = None,
        additional_params: Optional[dict[str, Any]] = None,  # noqa: ARG002
    ) -> tuple[np.ndarray, Optional[Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]]:
        return np.copy(image), boxes

    def get_config(self) -> dict[str, Any]:
        return {"param_1": self.param_1, "param_2": self.param_2}


class TestStepPerturbImageFactory:
    @pytest.mark.parametrize(
        ("perturber", "theta_key", "theta_value"),
        [
            (DummyPerturber, "param_1", 1.0),
            (DummyPerturber, "param_2", 3.0),
        ],
    )
    def test_iteration(
        self,
        snapshot: SnapshotAssertion,
        perturber: type[PerturbImage],
        theta_key: str,
        theta_value: float,
    ) -> None:
        """Ensure factory can be iterated upon and the varied parameter matches expectations."""
        factory = OneStepPerturbImageFactory(perturber=perturber, theta_key=theta_key, theta_value=theta_value)
        assert len(factory) == 1
        assert factory[0].get_config() == snapshot

    @pytest.mark.parametrize(
        ("perturber", "theta_key", "theta_value"),
        [(DummyPerturber, "param_1", 1.0), (DummyPerturber, "param_2", 3.0)],
    )
    def test_configuration(self, perturber: type[PerturbImage], theta_key: str, theta_value: float) -> None:
        """Test configuration stability."""
        inst = OneStepPerturbImageFactory(perturber=perturber, theta_key=theta_key, theta_value=theta_value)
        for i in configuration_test_helper(inst):
            assert i.perturber == perturber
            assert i.theta_key == theta_key
            assert i.theta_value == theta_value
            assert i.stop == theta_value + 0.1
            assert i.step == 1.0
            assert i.to_int
            assert i.theta_value in i.thetas
            assert i.stop not in i.thetas

    @pytest.mark.parametrize(
        ("perturber", "theta_key", "theta_value"),
        [(DummyPerturber, "param_1", 1.0), (DummyPerturber, "param_2", 3.0)],
    )
    def test_hydration(
        self,
        tmp_path: Path,
        perturber: type[PerturbImage],
        theta_key: str,
        theta_value: float,
    ) -> None:
        """Test configuration hydration using from_config_dict."""
        original_factory = OneStepPerturbImageFactory(perturber=perturber, theta_key=theta_key, theta_value=theta_value)

        original_factory_config = original_factory.get_config()

        config_file_path = tmp_path / "config.json"
        with open(str(config_file_path), "w") as f:
            json.dump(to_config_dict(original_factory), f)

        with open(str(config_file_path)) as config_file:
            config = json.load(config_file)
            hydrated_factory = from_config_dict(config, PerturbImageFactory.get_impls())
            hydrated_factory_config = hydrated_factory.get_config()

            assert original_factory_config == hydrated_factory_config
