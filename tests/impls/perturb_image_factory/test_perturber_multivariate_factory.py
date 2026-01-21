"""Tests for PerturberMultivariateFactory.

PerturberMultivariateFactory creates perturbers with theta values as a cartesian
product of multiple parameter sequences. Unlike other factories, it varies
multiple parameters simultaneously using theta_keys (plural).

Note: This class inherits from _TestPerturbImageFactory but overrides several
tests because:
    1. Uses theta_keys (list) instead of theta_key (str)
    2. Produces cartesian product combinations rather than linear sequences
    3. Has different thetas structure (list of lists vs single list)
    4. Has additional perturber_kwargs parameter

Test Cases (in addition to shared base class tests):
    Iteration (Valid)
        - Single key produces linear sequence
        - Two keys produce cartesian product (2x2)
        - Three keys produce cartesian product (2x2x2)

    Iteration (Empty)
        - Empty theta values [[]] produces empty factory

    Indexing
        - First index returns correct combination
        - Last index returns correct combination
        - Beyond bounds raises IndexError
        - Negative index raises IndexError

    Input Validation
        - Empty theta_keys raises ValueError
        - Mismatched theta_keys/thetas lengths raises ValueError

    perturber_kwargs Parameter
        - perturber_kwargs are passed to created perturbers
        - Theta values override perturber_kwargs for same key

    Length / Cartesian Product
        - Length equals number of values for single key
        - Length equals product of all value list lengths

    theta_key Property
        - Returns "params" (fixed value for multivariate)

    Edge Cases
        - Single value per key (no variation) works
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any

import pytest
from typing_extensions import override

from nrtk.impls.perturb_image_factory import PerturberMultivariateFactory
from tests.fakes import FakePerturber
from tests.impls.perturb_image_factory import _TestPerturbImageFactory


class TestPerturberMultivariateFactory(_TestPerturbImageFactory):
    """Tests for PerturberMultivariateFactory. See module docstring for test cases."""

    default_factory_kwargs: dict[str, Any] = {
        "theta_keys": ["param1", "param2"],
        "thetas": [[1, 3], [2, 4]],
    }

    @override
    def _make_factory(self, **kwargs: Any) -> PerturberMultivariateFactory:
        """Create a factory with FakePerturber pre-filled."""
        return PerturberMultivariateFactory(perturber=FakePerturber, **kwargs)

    # ========================= Iteration (Valid) ==========================
    # Override: multivariate uses theta_keys (plural) and cartesian products

    @pytest.mark.parametrize(
        ("factory_kwargs", "expected"),
        [
            pytest.param(
                {"theta_keys": ["param1"], "thetas": [[1, 2, 3]]},
                [(1,), (2,), (3,)],
                id="single key",
            ),
            pytest.param(
                {"theta_keys": ["param1", "param2"], "thetas": [[1, 3], [2, 4]]},
                [(1, 2), (1, 4), (3, 2), (3, 4)],
                id="cartesian product 2x2",
            ),
            pytest.param(
                {"theta_keys": ["param1", "param2", "extra"], "thetas": [[1, 2], [3, 4], [5, 6]]},
                [
                    (1, 3, 5),
                    (1, 3, 6),
                    (1, 4, 5),
                    (1, 4, 6),
                    (2, 3, 5),
                    (2, 3, 6),
                    (2, 4, 5),
                    (2, 4, 6),
                ],
                id="cartesian product 2x2x2",
            ),
        ],
    )
    @override
    def test_iteration_valid(
        self,
        factory_kwargs: dict[str, Any],
        expected: Sequence[tuple[Any, ...]],
    ) -> None:
        """Iteration produces correct cartesian product of theta values."""
        factory = self._make_factory(**factory_kwargs)
        theta_keys = factory_kwargs["theta_keys"]

        assert len(list(factory)) == len(expected)
        for perturber, expected_vals in zip(factory, expected, strict=True):
            assert isinstance(perturber, FakePerturber)
            config = perturber.get_config()
            for key, expected_val in zip(theta_keys, expected_vals, strict=True):
                assert config[key] == expected_val

    # ========================= Iteration (Empty) ==========================

    @pytest.mark.parametrize(
        "empty_factory_kwargs",
        [
            pytest.param(
                {"theta_keys": ["param1"], "thetas": [[]]},
                id="empty thetas",
            ),
        ],
    )
    @override
    def test_iteration_empty(self, empty_factory_kwargs: dict[str, Any]) -> None:
        super().test_iteration_empty(empty_factory_kwargs=empty_factory_kwargs)

    # ============================== Indexing ==============================
    # Multivariate checks multiple keys per perturber, different signature than base

    @pytest.mark.parametrize(
        ("idx", "expected_vals", "expectation"),
        [
            pytest.param(0, (1, 2), does_not_raise(), id="first"),
            pytest.param(3, (3, 4), does_not_raise(), id="last"),
            pytest.param(4, None, pytest.raises(IndexError), id="out of bounds"),
            pytest.param(-1, None, pytest.raises(IndexError), id="negative"),
        ],
    )
    def test_indexing_cartesian(
        self,
        idx: int,
        expected_vals: tuple[int, ...] | None,
        expectation: AbstractContextManager,
    ) -> None:
        """Indexing returns the correct perturber with cartesian product values."""
        factory = self._make_factory(**self.default_factory_kwargs)
        theta_keys = self.default_factory_kwargs["theta_keys"]
        with expectation:
            config = factory[idx].get_config()
            assert expected_vals is not None  # For type narrowing
            for key, expected in zip(theta_keys, expected_vals, strict=False):
                assert config[key] == expected

    @pytest.mark.skip(reason="Multivariate uses theta_keys (multiple), different signature")
    @override
    def test_indexing(
        self,
        idx: int,
        expected_val: float | None,
        expectation: AbstractContextManager,
    ) -> None:
        pass  # pragma: no cover

    # =========================== Property Tests ===========================
    # Override: len(factory) is cartesian product, not len(thetas)

    def test_len_is_cartesian_product(self) -> None:
        """len(factory) equals product of all theta list lengths."""
        factory = self._make_factory(**self.default_factory_kwargs)
        # default_factory_kwargs has thetas=[[1, 3], [2, 4]] -> 2 * 2 = 4
        assert len(factory) == 4

    @pytest.mark.skip(reason="Multivariate len is cartesian product, not len(thetas)")
    @override
    def test_len_matches_thetas_length(self) -> None:
        pass  # pragma: no cover

    # ========================== Input Validation ==========================

    def test_rejects_empty_theta_keys(self) -> None:
        """Empty theta_keys raises ValueError."""
        with pytest.raises(ValueError, match=r"theta_keys.*empty|at least one"):
            self._make_factory(theta_keys=list(), thetas=list())

    def test_rejects_mismatched_theta_keys_and_thetas_length(self) -> None:
        """Mismatched theta_keys/thetas lengths raises ValueError."""
        with pytest.raises(ValueError, match=r"length|must have.*same"):
            self._make_factory(
                theta_keys=["param1", "param2"],
                thetas=[[1, 2]],  # only one list, but two keys
            )

    # ===================== perturber_kwargs Parameter =====================

    def test_perturber_kwargs_passed_to_perturber(self) -> None:
        """perturber_kwargs are passed to created perturbers."""
        factory = self._make_factory(
            theta_keys=["param1"],
            thetas=[[1, 2]],
            perturber_kwargs={"param2": 99},
        )
        perturber = factory[0]
        config = perturber.get_config()
        assert config["param1"] == 1
        assert config["param2"] == 99

    def test_theta_values_override_perturber_kwargs(self) -> None:
        """Theta values override perturber_kwargs for same key."""
        factory = self._make_factory(
            theta_keys=["param1"],
            thetas=[[10, 20]],
            perturber_kwargs={"param1": 999},  # should be overridden
        )
        perturber = factory[0]
        config = perturber.get_config()
        assert config["param1"] == 10  # theta value, not perturber_kwargs

    # ========================= theta_key Property =========================

    def test_theta_key_returns_params(self) -> None:
        """theta_key property returns "params" (fixed value for multivariate)."""
        factory = self._make_factory(**self.default_factory_kwargs)
        assert factory.theta_key == "params"

    # ============================= Edge Cases =============================

    def test_single_value_per_key(self) -> None:
        """Single value per key (no variation) works."""
        factory = self._make_factory(
            theta_keys=["param1", "param2"],
            thetas=[[1], [2]],
        )
        assert len(factory) == 1
        config = factory[0].get_config()
        assert config["param1"] == 1
        assert config["param2"] == 2
