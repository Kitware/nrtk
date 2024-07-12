import numpy as np
import pytest
from contextlib import nullcontext as does_not_raise
from smqtk_core.configuration import configuration_test_helper
from typing import Any, ContextManager, Dict, Optional, Tuple, Type

from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.impls.perturb_image_factory.generic.linspace_step import LinSpacePerturbImageFactory


class DummyFloatPerturber(PerturbImage):
    def __init__(self, param1: float = 1, param2: float = 2):
        self.param1 = param1
        self.param2 = param2

    def perturb(
        self,
        image: np.ndarray,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:  # pragma: no cover
        return np.copy(image)

    def get_config(self) -> Dict[str, Any]:
        return {
            "param1": self.param1,
            "param2": self.param2
        }


class TestFloatStepPertubImageFactory:
    @pytest.mark.parametrize("perturber, theta_key, start, stop, step, expected", [
        (DummyFloatPerturber, "param1", 1, 3, 4, (1, 1.5, 2, 2.5)),
        (DummyFloatPerturber, "param2", 3, 9, 2, (3, 6)),
        (DummyFloatPerturber, "param1", 4, 4, 1, ())
    ])
    def test_iteration(
        self,
        perturber: Type[PerturbImage],
        theta_key: str,
        start: int,
        stop: int,
        step: int,
        expected: Tuple[int, ...]
    ) -> None:
        """
        Ensure factory can be iterated upon and the varied parameter matches expectations.
        """
        factory = LinSpacePerturbImageFactory(
            perturber=perturber,
            theta_key=theta_key,
            start=start,
            stop=stop,
            step=step
        )
        assert len(expected) == len(factory)
        for idx, p in enumerate(factory):
            print(p.get_config(), expected)
            assert p.get_config()[theta_key] == expected[idx]

    @pytest.mark.parametrize("perturber, theta_key, start, stop, step, idx, expected_val, expectation", [
        (DummyFloatPerturber, "param1", 1, 6, 10, 0, 1, does_not_raise()),
        (DummyFloatPerturber, "param1", 1, 6, 10, 3, 2.5, does_not_raise()),
        (DummyFloatPerturber, "param1", 1, 6, 2, 3, -1, pytest.raises(IndexError)),
        (DummyFloatPerturber, "param1", 1, 6, 2, -1, -1, pytest.raises(IndexError)),
        (DummyFloatPerturber, "param1", 4, 4, 1, 0, -1, pytest.raises(IndexError))
    ], ids=["first idx", "last idx", "idx == len", "neg idx", "empty iter"])
    def test_indexing(
        self,
        perturber: Type[PerturbImage],
        theta_key: str,
        start: int,
        stop: int,
        step: int,
        idx: int,
        expected_val: int,
        expectation: ContextManager
    ) -> None:
        """
        Ensure it is possible to access a perturber instance via indexing.
        """
        factory = LinSpacePerturbImageFactory(
            perturber=perturber,
            theta_key=theta_key,
            start=start,
            stop=stop,
            step=step
        )
        with expectation:
            print(idx, factory[idx].get_config(), expected_val)
            assert factory[idx].get_config()[theta_key] == expected_val

    @pytest.mark.parametrize("perturber, theta_key, start, stop, step", [
        (DummyFloatPerturber, "param1", 1.0, 5.0, 2),
        (DummyFloatPerturber, "param2", 3.0, 9.0, 3)
    ])
    def test_configuration(
        self,
        perturber: Type[PerturbImage],
        theta_key: str,
        start: float,
        stop: float,
        step: int
    ) -> None:
        """
        Test configuration stability.
        """
        inst = LinSpacePerturbImageFactory(
            perturber=perturber,
            theta_key=theta_key,
            start=start,
            stop=stop,
            step=step
        )
        for i in configuration_test_helper(inst):
            assert i.perturber == perturber
            assert i.theta_key == theta_key
            assert i.start == start
            assert i.stop == stop
            assert i.step == step
            assert start in i.thetas and stop not in i.thetas

    @pytest.mark.parametrize("kwargs, expectation", [
        ({"perturber": DummyFloatPerturber, "theta_key": "param1", "start": 1, "stop": 2}, does_not_raise()),
        ({"perturber": DummyFloatPerturber(1, 2), "theta_key": "param2", "start": 1, "stop": 2},
            pytest.raises(TypeError, match=r"Passed a perturber instance, expected type")),
    ])
    def test_configuration_bounds(self, kwargs: Dict[str, Any], expectation: ContextManager) -> None:
        """
        Test that an exception is properly raised (or not) based on argument value.
        """
        with expectation:
            LinSpacePerturbImageFactory(**kwargs)
