"""Tests for PerturbImageFactory interface.

This module tests the abstract PerturbImageFactory interface behavior using
PerturberFakeFactory as the concrete implementation, inheriting shared test
cases from PerturberFactoryMixin.

Test Cases (inherited from PerturberFactoryMixin):
    - Iteration (valid/empty)
    - Indexing
    - Repeatability
    - Configuration hydration
    - Input validation (rejects perturber instance)
    - Property tests (len, theta_key, thetas, getitem matches iteration)

Test Cases (interface-specific):
    - from_config resolves perturber type string to class
    - from_config rejects invalid perturber type string
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any

import pytest
from typing_extensions import override

from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from tests.fakes import FakePerturber, PerturberFakeFactory
from tests.impls.perturb_image_factory import PerturberFactoryMixin


@pytest.mark.core
class TestPerturbImageFactory(PerturberFactoryMixin):
    """Tests for PerturbImageFactory interface."""

    default_factory_kwargs = {
        "theta_key": "param1",
        "theta_values": [1, 2, 3],
    }

    @override
    def _make_factory(self, **kwargs: Any) -> PerturbImageFactory:
        """Create a PerturberFakeFactory for testing."""
        return PerturberFakeFactory(perturber=FakePerturber, **kwargs)

    # =========================== Skipped Tests ============================

    @pytest.mark.skip(reason="PerturberFakeFactory is not a registered plugin")
    @override
    def test_discoverability(self) -> None:
        pass  # pragma: no cover

    # ==================== Parametrized Inherited Tests ====================

    @pytest.mark.parametrize(
        ("factory_kwargs", "expected"),
        [
            pytest.param(
                {"theta_key": "param1", "theta_values": [1, 2, 3]},
                [1, 2, 3],
                id="three values",
            ),
            pytest.param(
                {"theta_key": "param1", "theta_values": [10]},
                [10],
                id="single value",
            ),
        ],
    )
    @override
    def test_iteration_valid(self, factory_kwargs: dict[str, Any], expected: Sequence[Any]) -> None:
        super().test_iteration_valid(factory_kwargs=factory_kwargs, expected=expected)

    @pytest.mark.parametrize(
        "empty_factory_kwargs",
        [
            pytest.param(
                {"theta_key": "param1", "theta_values": []},
                id="empty theta_values",
            ),
        ],
    )
    @override
    def test_iteration_empty(self, empty_factory_kwargs: dict[str, Any]) -> None:
        super().test_iteration_empty(empty_factory_kwargs=empty_factory_kwargs)

    @pytest.mark.parametrize(
        ("idx", "expected_val", "expectation"),
        [
            pytest.param(0, 1, does_not_raise(), id="first"),
            pytest.param(2, 3, does_not_raise(), id="last"),
            pytest.param(3, None, pytest.raises(IndexError), id="out of bounds positive"),
            pytest.param(-1, 3, does_not_raise(), id="negative -1 (last)"),
            pytest.param(-3, 1, does_not_raise(), id="negative -3 (first)"),
            pytest.param(-4, None, pytest.raises(IndexError), id="out of bounds negative"),
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

    # ====================== Interface-Specific Tests ======================

    def test_from_config_resolves_perturber_type(self) -> None:
        """from_config resolves perturber type string to class."""
        config = {
            "perturber": FakePerturber.get_type_string(),
            "theta_key": "param1",
            "theta_values": [1, 2, 3],
        }
        factory = PerturberFakeFactory.from_config(config)
        assert factory.perturber is FakePerturber

    def test_from_config_rejects_invalid_perturber(self) -> None:
        """from_config rejects invalid perturber type string."""
        config = {
            "perturber": "not.a.real.Perturber",
            "theta_key": "param1",
            "theta_values": [1],
        }
        with pytest.raises(ValueError, match=r"not a valid perturber"):
            PerturberFakeFactory.from_config(config)
