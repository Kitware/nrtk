"""Tests for PerturberStepFactory.

PerturberStepFactory creates perturbers with theta values at regular step
intervals over a range, using numpy.arange-like semantics.

Test Cases (in addition to shared base class tests):
    Iteration (Valid)
        - Integer steps with to_int=True
        - Float steps with to_int=False

    Iteration (Empty)
        - start == stop produces empty factory

    Indexing
        - Indexing empty factory raises IndexError

    Edge Cases (parametrized)
        - Step larger than range produces single value
        - Negative step with descending range works
        - Fractional step count (non-integer division) handled correctly
        - Negative value ranges work correctly

    to_int Parameter (parametrized)
        - to_int=True returns integer theta values
        - to_int=False returns float theta values
        - to_int=True truncates fractional values
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any

import numpy as np
import pytest
from typing_extensions import override

from nrtk.impls.perturb_image_factory import PerturberStepFactory
from tests.fakes import FakePerturber
from tests.impls.perturb_image_factory import PerturberFactoryMixin


class TestPerturberStepFactory(PerturberFactoryMixin):
    """Tests for PerturberStepFactory. See module docstring for test cases."""

    default_factory_kwargs = {
        "theta_key": "param1",
        "start": 1.0,
        "stop": 6.0,
        "step": 2.0,
        "to_int": True,
    }

    @override
    def _make_factory(self, **kwargs: Any) -> PerturberStepFactory:
        """Create a factory with FakePerturber pre-filled."""
        return PerturberStepFactory(perturber=FakePerturber, **kwargs)

    # ========================= Iteration (Valid) ==========================

    @pytest.mark.parametrize(
        ("factory_kwargs", "expected"),
        [
            pytest.param(
                {
                    "theta_key": "param1",
                    "start": 1.0,
                    "stop": 6.0,
                    "step": 2.0,
                    "to_int": True,
                },
                [1.0, 3.0, 5.0],
                id="integer steps",
            ),
            pytest.param(
                {
                    "theta_key": "param1",
                    "start": 3.0,
                    "stop": 9.0,
                    "step": 1.5,
                    "to_int": False,
                },
                [3.0, 4.5, 6.0, 7.5],
                id="float steps",
            ),
        ],
    )
    @override
    def test_iteration_valid(self, factory_kwargs: dict[str, Any], expected: Sequence[Any]) -> None:
        super().test_iteration_valid(factory_kwargs=factory_kwargs, expected=expected)

    # ========================= Iteration (Empty) ==========================

    @pytest.mark.parametrize(
        "empty_factory_kwargs",
        [
            pytest.param(
                {"theta_key": "param1", "start": 4.0, "stop": 4.0, "step": 1.0},
                id="start equals stop",
            ),
        ],
    )
    @override
    def test_iteration_empty(self, empty_factory_kwargs: dict[str, Any]) -> None:
        super().test_iteration_empty(empty_factory_kwargs=empty_factory_kwargs)

    # ============================== Indexing ==============================

    @pytest.mark.parametrize(
        ("idx", "expected_val", "expectation"),
        [
            pytest.param(0, 1, does_not_raise(), id="first"),
            pytest.param(2, 5, does_not_raise(), id="last"),
            pytest.param(3, None, pytest.raises(IndexError), id="out of bounds"),
            pytest.param(-1, None, pytest.raises(IndexError), id="negative"),
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

    def test_indexing_empty(self) -> None:
        """Indexing empty factory raises IndexError."""
        factory = self._make_factory(
            theta_key="param1",
            start=4.0,
            stop=4.0,
            step=1.0,
        )
        with pytest.raises(IndexError):
            _ = factory[0]

    # ============================= Edge Cases =============================

    @pytest.mark.parametrize(
        ("factory_kwargs", "expected_len", "expected_thetas"),
        [
            pytest.param(
                {"theta_key": "param1", "start": 0.0, "stop": 0.5, "step": 1.0},
                1,
                [0.0],
                id="step larger than range",
            ),
            pytest.param(
                {"theta_key": "param1", "start": 5.0, "stop": 0.0, "step": -1.0},
                5,
                [5.0, 4.0, 3.0, 2.0, 1.0],
                id="negative step descending",
            ),
            pytest.param(
                {"theta_key": "param1", "start": 0.0, "stop": 1.0, "step": 0.3},
                4,
                [0.0, 0.3, 0.6, 0.9],
                id="fractional step count",
            ),
            pytest.param(
                {"theta_key": "param1", "start": -2.0, "stop": 2.0, "step": 1.0},
                4,
                [-2.0, -1.0, 0.0, 1.0],
                id="negative values",
            ),
        ],
    )
    def test_edge_cases(
        self,
        factory_kwargs: dict[str, Any],
        expected_len: int,
        expected_thetas: list[float],
    ) -> None:
        """Edge case ranges produce correct theta values."""
        factory = self._make_factory(**factory_kwargs)
        assert len(factory) == expected_len
        for actual, exp in zip(factory.thetas, expected_thetas, strict=True):
            assert np.isclose(actual, exp, atol=1e-4)

    # ========================== to_int Parameter ==========================

    @pytest.mark.parametrize(
        ("to_int", "start", "expected_type", "expected_values"),
        [
            pytest.param(True, 1.0, int, [1, 2, 3, 4], id="to_int=True returns integers"),
            pytest.param(False, 1.0, float, [1.0, 2.0, 3.0, 4.0], id="to_int=False returns floats"),
            pytest.param(True, 1.5, int, [1, 2, 3, 4], id="to_int=True truncates fractional"),
        ],
    )
    def test_to_int_returns_correct_type(
        self,
        to_int: bool,
        start: float,
        expected_type: type,
        expected_values: list,
    ) -> None:
        """to_int parameter controls whether theta values are int or float."""
        factory = self._make_factory(
            theta_key="param1",
            start=start,
            stop=5.0,
            step=1.0,
            to_int=to_int,
        )
        thetas = factory.thetas
        assert all(isinstance(t, expected_type) for t in thetas)
        assert np.allclose(thetas, expected_values)
