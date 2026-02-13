"""Tests for PerturberOneStepFactory.

PerturberOneStepFactory is a convenience factory that creates exactly one
perturber with a single theta value. It extends PerturberStepFactory.

Note: This factory always produces exactly one perturber, so there are no
empty factory cases.

Test Cases (in addition to shared base class tests):
    Iteration (Valid)
        - Single positive value
        - Zero value
        - Negative value

    to_int Parameter (parametrized)
        - to_int=True returns integer theta value
        - to_int=False returns float theta value
        - to_int=True truncates fractional value
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any

import numpy as np
import pytest
from typing_extensions import override

from nrtk.impls.perturb_image_factory import PerturberOneStepFactory
from tests.fakes import FakePerturber
from tests.impls.perturb_image_factory import PerturberFactoryMixin


@pytest.mark.core
class TestPerturberOneStepFactory(PerturberFactoryMixin):
    """Tests for PerturberOneStepFactory. See module docstring for test cases."""

    default_factory_kwargs = {
        "theta_key": "param1",
        "theta_value": 5.0,
    }

    @override
    def _make_factory(self, **kwargs: Any) -> PerturberOneStepFactory:
        """Create a factory with FakePerturber pre-filled."""
        return PerturberOneStepFactory(perturber=FakePerturber, **kwargs)

    # ========================= Iteration (Valid) ==========================

    @pytest.mark.parametrize(
        ("factory_kwargs", "expected"),
        [
            pytest.param(
                {"theta_key": "param1", "theta_value": 5.0},
                [5.0],
                id="single value",
            ),
            pytest.param(
                {"theta_key": "param1", "theta_value": 0.0},
                [0.0],
                id="zero value",
            ),
            pytest.param(
                {"theta_key": "param1", "theta_value": -5.0},
                [-5.0],
                id="negative value",
            ),
        ],
    )
    @override
    def test_iteration_valid(self, factory_kwargs: dict[str, Any], expected: Sequence[Any]) -> None:
        super().test_iteration_valid(factory_kwargs=factory_kwargs, expected=expected)

    # ========================= Iteration (Empty) ==========================

    @pytest.mark.skip(reason="OneStepFactory always produces exactly one perturber; no empty cases possible")
    @override
    def test_iteration_empty(self, empty_factory_kwargs: dict[str, Any]) -> None:  # pragma: no cover
        pass

    # ============================== Indexing ==============================

    @pytest.mark.parametrize(
        ("idx", "expected_val", "expectation"),
        [
            pytest.param(0, 5.0, does_not_raise(), id="index 0"),
            pytest.param(1, None, pytest.raises(IndexError), id="out of bounds positive"),
            pytest.param(-1, 5.0, does_not_raise(), id="negative -1 (last/only)"),
            pytest.param(-2, None, pytest.raises(IndexError), id="out of bounds negative"),
        ],
    )
    @override
    def test_indexing(
        self,
        idx: int,
        expected_val: float | None,
        expectation: AbstractContextManager,
    ) -> None:
        super().test_indexing(idx=idx, expected_val=expected_val, expectation=expectation)

    # ========================== to_int Parameter ==========================

    @pytest.mark.parametrize(
        ("to_int", "theta_value", "expected_type", "expected_value"),
        [
            pytest.param(True, 5.0, int, 5, id="to_int=True returns integer"),
            pytest.param(False, 5.0, float, 5.0, id="to_int=False returns float"),
            pytest.param(True, 5.7, int, 5, id="to_int=True truncates fractional"),
        ],
    )
    def test_to_int_returns_correct_type(
        self,
        to_int: bool,
        theta_value: float,
        expected_type: type,
        expected_value: float,
    ) -> None:
        """to_int parameter controls whether theta values are int or float."""
        factory = self._make_factory(
            theta_key="param1",
            theta_value=theta_value,
            to_int=to_int,
        )
        thetas = factory.thetas
        assert len(thetas) == 1
        assert isinstance(thetas[0], expected_type)
        assert np.isclose(thetas[0], expected_value)
