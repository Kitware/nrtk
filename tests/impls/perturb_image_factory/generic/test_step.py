from __future__ import annotations

import json
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from smqtk_core.configuration import (
    configuration_test_helper,
    from_config_dict,
    to_config_dict,
)
from syrupy.assertion import SnapshotAssertion

from nrtk.impls.perturb_image.pybsm.detector_otf_perturber import DetectorOTFPerturber
from nrtk.impls.perturb_image.pybsm.jitter_otf_perturber import JitterOTFPerturber
from nrtk.impls.perturb_image.pybsm.turbulence_aperture_otf_perturber import TurbulenceApertureOTFPerturber
from nrtk.impls.perturb_image_factory.generic.step import StepPerturbImageFactory
from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from tests.impls.perturb_image.test_perturber_utils import pybsm_perturber_assertions
from tests.test_utils import CustomFloatSnapshotExtension

DATA_DIR = Path(__file__).parents[3] / "data"
INPUT_IMG_FILE_PATH = "./examples/pybsm/data/M-41 Walker Bulldog (USA) width 319cm height 272cm.tiff"


@pytest.fixture()  # noqa:PT001
def snapshot_custom(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(lambda: CustomFloatSnapshotExtension())


class DummyPerturber(PerturbImage):
    def __init__(self, param_1: int = 1, param_2: int = 2) -> None:
        self.param_1 = param_1
        self.param_2 = param_2

    def perturb(
        self,
        image: np.ndarray,
        _: dict[str, Any] | None = None,
    ) -> np.ndarray:  # pragma: no cover
        return np.copy(image)

    def get_config(self) -> dict[str, Any]:
        return {"param_1": self.param_1, "param_2": self.param_2}


class TestStepPerturbImageFactory:
    @pytest.mark.parametrize(
        ("perturber", "theta_key", "start", "stop", "step", "to_int", "expected"),
        [
            (DummyPerturber, "param_1", 1.0, 6.0, 2.0, True, (1.0, 3.0, 5.0)),
            (DummyPerturber, "param_2", 3.0, 9.0, 3.0, True, (3.0, 6.0)),
            (DummyPerturber, "param_1", 3.0, 9.0, 1.5, False, (3.0, 4.5, 6.0, 7.5)),
            (DummyPerturber, "param_1", 4.0, 4.0, 1.0, False, ()),
        ],
    )
    def test_iteration(
        self,
        perturber: type[PerturbImage],
        theta_key: str,
        start: float,
        stop: float,
        step: float,
        to_int: bool,
        expected: tuple[float, ...],
    ) -> None:
        """Ensure factory can be iterated upon and the varied parameter matches expectations."""
        factory = StepPerturbImageFactory(
            perturber=perturber,
            theta_key=theta_key,
            start=start,
            stop=stop,
            step=step,
            to_int=to_int,
        )
        assert len(expected) == len(factory)
        for idx, p in enumerate(factory):
            assert p.get_config()[theta_key] == expected[idx]

    @pytest.mark.parametrize(
        (
            "perturber",
            "theta_key",
            "start",
            "stop",
            "step",
            "idx",
            "expected_val",
            "expectation",
        ),
        [
            (DummyPerturber, "param_1", 1.0, 6.0, 2.0, 0, 1, does_not_raise()),
            (DummyPerturber, "param_1", 1.0, 6.0, 2.0, 2, 5, does_not_raise()),
            (DummyPerturber, "param_1", 1.0, 6.0, 2.0, 3, -1, pytest.raises(IndexError)),
            (DummyPerturber, "param_1", 1.0, 6.0, 2.0, -1, -1, pytest.raises(IndexError)),
            (DummyPerturber, "param_1", 4.0, 4.0, 1.0, 0, -1, pytest.raises(IndexError)),
        ],
        ids=["first idx", "last idx", "idx == len", "neg idx", "empty iter"],
    )
    def test_indexing(
        self,
        perturber: type[PerturbImage],
        theta_key: str,
        start: float,
        stop: float,
        step: float,
        idx: int,
        expected_val: float,
        expectation: AbstractContextManager,
    ) -> None:
        """Ensure it is possible to access a perturber instance via indexing."""
        factory = StepPerturbImageFactory(perturber=perturber, theta_key=theta_key, start=start, stop=stop, step=step)
        with expectation:
            assert factory[idx].get_config()[theta_key] == expected_val

    @pytest.mark.parametrize(
        ("perturber", "theta_key", "start", "stop", "step", "to_int"),
        [(DummyPerturber, "param_1", 1.0, 5.0, 2.0, False), (DummyPerturber, "param_2", 3.0, 9.0, 3.0, True)],
    )
    def test_configuration(
        self,
        perturber: type[PerturbImage],
        theta_key: str,
        start: float,
        stop: float,
        step: float,
        to_int: bool,
    ) -> None:
        """Test configuration stability."""
        inst = StepPerturbImageFactory(
            perturber=perturber,
            theta_key=theta_key,
            start=start,
            stop=stop,
            step=step,
            to_int=to_int,
        )
        for i in configuration_test_helper(inst):
            assert i.perturber == perturber
            assert i.theta_key == theta_key
            assert i.start == start
            assert i.stop == stop
            assert i.step == step
            assert start in i.thetas
            assert stop not in i.thetas

    @pytest.mark.parametrize(
        ("kwargs", "expectation"),
        [
            (
                {
                    "perturber": DummyPerturber,
                    "theta_key": "param_1",
                    "start": 1.0,
                    "stop": 2.0,
                },
                does_not_raise(),
            ),
            (
                {
                    "perturber": DummyPerturber(1, 2),
                    "theta_key": "param_2",
                    "start": 1.0,
                    "stop": 2.0,
                },
                pytest.raises(TypeError, match=r"Passed a perturber instance, expected type"),
            ),
        ],
    )
    def test_configuration_bounds(self, kwargs: dict[str, Any], expectation: AbstractContextManager) -> None:
        """Test that an exception is properly raised (or not) based on argument value."""
        with expectation:
            StepPerturbImageFactory(**kwargs)

    @pytest.mark.parametrize(
        ("perturber", "theta_key", "start", "stop", "step"),
        [(DummyPerturber, "param_1", 1.0, 5.0, 2.0), (DummyPerturber, "param_2", 3.0, 9.0, 3.0)],
    )
    def test_hydration(
        self,
        tmp_path: Path,
        perturber: type[PerturbImage],
        theta_key: str,
        start: float,
        stop: float,
        step: float,
    ) -> None:
        """Test configuration hydration using from_config_dict."""
        original_factory = StepPerturbImageFactory(
            perturber=perturber,
            theta_key=theta_key,
            start=start,
            stop=stop,
            step=step,
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

    @pytest.mark.parametrize(
        ("config_file_name", "expectation"),
        [
            (
                "nrtk_brightness_config.json",
                does_not_raise(),
            ),
            (
                "nrtk_bad_step_config.json",
                pytest.raises(ValueError, match=r"not a perturber is not a valid perturber."),
            ),
        ],
    )
    def test_hydration_bounds(self, config_file_name: str, expectation: AbstractContextManager) -> None:
        """Test that an exception is properly raised (or not) based on argument value."""
        with expectation, open(str(DATA_DIR / config_file_name)) as config_file:
            config = json.load(config_file)
            from_config_dict(config, PerturbImageFactory.get_impls())

    @pytest.mark.parametrize(
        ("perturber", "modifying_param", "modifying_val", "theta_key", "start", "stop", "step"),
        [
            (JitterOTFPerturber, "s_y", 0, "s_x", 2e-3, 6e-3, 1e-3),
            (DetectorOTFPerturber, "w_x", 0, "w_y", 3e-3, 9e-3, 1e-3),
            (TurbulenceApertureOTFPerturber, "altitude", 250, "D", 40e-5, 40e-3, 66e-4),
        ],
    )
    def test_perturb_instance_modification(
        self,
        snapshot_custom: SnapshotAssertion,
        perturber: type[PerturbImage],
        modifying_param: str,
        modifying_val: float,
        theta_key: str,
        start: float,
        stop: float,
        step: float,
    ) -> None:
        """Test perturber instance modification for a perturber factory."""
        perturber_factory = StepPerturbImageFactory(
            perturber=perturber,
            theta_key=theta_key,
            start=start,
            stop=stop,
            step=step,
            to_int=False,
        )
        img = np.array(Image.open(INPUT_IMG_FILE_PATH))
        img_md = {"img_gsd": 3.19 / 160.0}
        for perturber in perturber_factory:
            setattr(perturber, modifying_param, modifying_val)
            out_img = pybsm_perturber_assertions(perturb=perturber, image=img, expected=None, additional_params=img_md)
            snapshot_custom.assert_match(out_img)
