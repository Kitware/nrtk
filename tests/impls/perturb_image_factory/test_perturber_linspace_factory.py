"""Tests for PerturberLinspaceFactory.

PerturberLinspaceFactory creates perturbers with theta values linearly spaced
over a range, using numpy.linspace semantics.

Test Cases (in addition to shared base class tests):
    Iteration (Valid)
        - 5 evenly spaced values from 0.0 to 1.0
        - 2 values (just endpoints)
        - Single value when start equals stop with num=1

    Iteration (Empty)
        - num=0 produces empty factory

    Endpoint Parameter
        - endpoint=True includes stop value
        - endpoint=False excludes stop value

    Edge Cases
        - start > stop produces descending range
        - Negative value ranges work correctly
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any

import numpy as np
import pytest
from typing_extensions import override

from nrtk.impls.perturb_image_factory import PerturberLinspaceFactory
from tests.fakes import FakePerturber
from tests.impls.perturb_image_factory import PerturberFactoryMixin


@pytest.mark.core
class TestPerturberLinspaceFactory(PerturberFactoryMixin):
    """Tests for PerturberLinspaceFactory. See module docstring for test cases."""

    default_factory_kwargs = {
        "theta_key": "param1",
        "start": 0.0,
        "stop": 1.0,
        "num": 5,
    }

    @override
    def _make_factory(self, **kwargs: Any) -> PerturberLinspaceFactory:
        """Create a factory with FakePerturber pre-filled."""
        return PerturberLinspaceFactory(perturber=FakePerturber, **kwargs)

    # ========================= Iteration (Valid) ==========================

    @pytest.mark.parametrize(
        ("factory_kwargs", "expected"),
        [
            pytest.param(
                {"theta_key": "param1", "start": 0.0, "stop": 1.0, "num": 5},
                [0.0, 0.25, 0.5, 0.75, 1.0],
                id="5 evenly spaced values",
            ),
            pytest.param(
                {"theta_key": "param1", "start": 0.0, "stop": 1.0, "num": 2},
                [0.0, 1.0],
                id="2 values (endpoints)",
            ),
            pytest.param(
                {"theta_key": "param1", "start": 5.0, "stop": 5.0, "num": 1},
                [5.0],
                id="single value",
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
                {"theta_key": "param1", "start": 0.0, "stop": 1.0, "num": 0},
                id="num=0",
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
            pytest.param(0, 0.0, does_not_raise(), id="first"),
            pytest.param(4, 1.0, does_not_raise(), id="last"),
            pytest.param(5, None, pytest.raises(IndexError), id="out of bounds positive"),
            pytest.param(-1, 1.0, does_not_raise(), id="negative -1 (last)"),
            pytest.param(-5, 0.0, does_not_raise(), id="negative -5 (first)"),
            pytest.param(-6, None, pytest.raises(IndexError), id="out of bounds negative"),
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
            start=0.0,
            stop=1.0,
            num=0,
        )
        with pytest.raises(IndexError):
            _ = factory[0]
        with pytest.raises(IndexError):
            _ = factory[-1]

    # ========================= Endpoint Parameter =========================

    @pytest.mark.parametrize(
        ("endpoint", "expected"),
        [
            pytest.param(True, [0.0, 0.5, 1.0], id="endpoint=True includes stop"),
            pytest.param(False, [0.0, 0.333333, 0.666666], id="endpoint=False excludes stop"),
        ],
    )
    def test_endpoint_parameter(self, endpoint: bool, expected: list[float]) -> None:
        """Endpoint parameter controls whether stop value is included.

        Verifies that:
        - endpoint=True includes stop value in output
        - endpoint=False excludes stop value from output
        """
        factory = self._make_factory(
            theta_key="param1",
            start=0.0,
            stop=1.0,
            num=3,
            endpoint=endpoint,
        )
        # Compare with tolerance for floating point
        for actual, exp in zip(factory.thetas, expected, strict=False):
            assert np.isclose(actual, exp, atol=1e-4)

    # ============================= Edge Cases =============================

    def test_start_greater_than_stop(self) -> None:
        """Start > stop produces descending range."""
        factory = self._make_factory(
            theta_key="param1",
            start=1.0,
            stop=0.0,
            num=3,
        )
        expected = [1.0, 0.5, 0.0]
        for actual, exp in zip(factory.thetas, expected, strict=False):
            assert np.isclose(actual, exp, atol=1e-4)

    def test_negative_values(self) -> None:
        """Negative value ranges work correctly."""
        factory = self._make_factory(
            theta_key="param1",
            start=-1.0,
            stop=1.0,
            num=5,
        )
        expected = [-1.0, -0.5, 0.0, 0.5, 1.0]
        for actual, exp in zip(factory.thetas, expected, strict=False):
            assert np.isclose(actual, exp, atol=1e-4)
