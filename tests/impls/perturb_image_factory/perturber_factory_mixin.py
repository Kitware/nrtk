"""Base test class for PerturbImageFactory implementations.

This module provides ``PerturberFactoryMixin``, an abstract base test class that
defines shared test cases for all PerturbImageFactory implementations. Concrete
factory test classes (e.g., TestPerturberLinspaceFactory, TestPerturberStepFactory)
inherit from this class to automatically run common tests.

Usage:
    Subclasses must define:
        - ``_make_factory(**kwargs)``: Create a factory instance for testing
        - ``default_factory_kwargs``: Default kwargs for creating a test factory

    Subclasses must parametrize:
        - ``test_iteration_valid``: With (factory_kwargs, expected) pairs
        - ``test_iteration_empty``: With empty_factory_kwargs (or skip if not applicable)
        - ``test_indexing``: With (idx, expected_val, expectation) tuples

Shared Test Cases:
    Plugin Discovery
        - Factory is discoverable via PerturbImageFactory.get_impls()

    Iteration (Valid)
        - Iteration produces perturbers with correct theta values
        - Each perturber is an instance of the specified perturber type
        - Number of perturbers matches expected count

    Iteration (Empty)
        - Edge case inputs produce empty factory with len() == 0
        - Iterating empty factory produces empty list

    Indexing
        - Index 0 returns first perturber with correct theta
        - Last valid index returns last perturber with correct theta
        - Index beyond bounds raises IndexError
        - Negative index raises IndexError

    Repeatability
        - Iterating factory multiple times produces identical results
        - Configs match between iterations

    Configuration Hydration
        - Factory survives get_config()/from_config() roundtrip
        - Reconstructed factory has same length as original
        - Factory survives JSON serialization/deserialization roundtrip
        - Reconstructed factory config matches original

    Input Validation
        - Passing perturber instance (not type) raises TypeError

    Property Tests
        - len(factory) equals len(factory.thetas)
        - theta_key property returns a string
        - thetas property returns a Sequence
        - factory[i] matches i-th item from iteration
"""

from __future__ import annotations

import json
from abc import abstractmethod
from collections.abc import Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import pytest
from smqtk_core.configuration import configuration_test_helper, from_config_dict, to_config_dict

from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from tests.fakes import FakePerturber
from tests.utils import deep_equals


class PerturberFactoryMixin:
    """Base test class for PerturbImageFactory implementations.

    See module docstring for full list of shared test cases and usage instructions.
    """

    default_factory_kwargs: dict[str, Any]

    @abstractmethod
    def _make_factory(self, **kwargs: Any) -> PerturbImageFactory:
        """Create a factory instance for testing. Subclasses must implement."""
        ...  # pragma: no cover

    # ========================== Plugin Discovery ==========================

    def test_discoverability(self) -> None:
        """Factory is discoverable via PerturbImageFactory.get_impls()."""
        factory = self._make_factory(**self.default_factory_kwargs)
        assert type(factory) in PerturbImageFactory.get_impls()

    # ========================= Iteration (Valid) ==========================

    def test_iteration_valid(self, factory_kwargs: dict[str, Any], expected: Sequence[Any]) -> None:
        """Iteration produces perturbers with correct theta values.

        Verifies that:
        - Number of perturbers matches expected count
        - Each perturber is an instance of the specified perturber type
        - Each perturber has the correct theta value in its config
        """
        factory = self._make_factory(**factory_kwargs)
        theta_key = factory_kwargs["theta_key"]

        assert len(list(factory)) == len(expected)
        for perturber, expected_val in zip(factory, expected, strict=True):
            assert isinstance(perturber, FakePerturber)
            assert perturber.get_config()[theta_key] == expected_val

    # ========================= Iteration (Empty) ==========================

    def test_iteration_empty(self, empty_factory_kwargs: dict[str, Any]) -> None:
        """Edge case inputs produce empty factory with no perturbers.

        Verifies that:
        - Factory length is 0
        - Iterating produces an empty list
        """
        factory = self._make_factory(**empty_factory_kwargs)
        assert len(factory) == 0
        assert list(factory) == list()

    # ============================== Indexing ==============================

    def test_indexing(
        self,
        idx: int,
        expected_val: float | None,
        expectation: AbstractContextManager,
    ) -> None:
        """Indexing returns the correct perturber or raises IndexError.

        Verifies that:
        - Index 0 returns first perturber with correct theta
        - Last valid index returns last perturber with correct theta
        - Index beyond bounds raises IndexError
        - Negative index raises IndexError
        """
        factory = self._make_factory(**self.default_factory_kwargs)
        theta_key = self.default_factory_kwargs["theta_key"]
        with expectation:
            assert factory[idx].get_config()[theta_key] == expected_val

    # =========================== Repeatability ============================

    def test_repeatability(self) -> None:
        """Iterating the factory multiple times produces identical results.

        Verifies that:
        - First and second iteration have same length
        - Perturber configs match between iterations
        """
        factory = self._make_factory(**self.default_factory_kwargs)

        first_pass = list(factory)
        second_pass = list(factory)

        assert len(first_pass) == len(second_pass)
        for p1, p2 in zip(first_pass, second_pass, strict=True):
            assert deep_equals(a=p1.get_config(), b=p2.get_config())

    # ====================== Configuration Hydration =======================

    def test_hydration(self, tmp_path: Path) -> None:
        """Factory survives configuration roundtrip via multiple methods.

        Verifies that:
        - Factory can be serialized and deserialized via configuration_test_helper
        - Reconstructed factory has same length as original
        - Factory can be serialized to JSON file and deserialized
        - Reconstructed factory config matches original config
        """
        factory = self._make_factory(**self.default_factory_kwargs)
        original_config = factory.get_config()

        # Test via configuration_test_helper
        for reconstructed in configuration_test_helper(factory):
            assert len(reconstructed) == len(factory)

        # Test via JSON file serialization
        config_file_path = tmp_path / "config.json"
        with open(config_file_path, "w") as f:
            json.dump(to_config_dict(factory), f)

        # Read back and hydrate
        with open(config_file_path) as f:
            config = json.load(f)
            hydrated_factory = from_config_dict(config=config, type_iter=PerturbImageFactory.get_impls())
            hydrated_config = hydrated_factory.get_config()

        assert deep_equals(a=original_config, b=hydrated_config)

    # ========================== Input Validation ==========================

    def test_rejects_perturber_instance(self) -> None:
        """Passing perturber instance (not type) raises TypeError."""
        impl_class = type(self._make_factory(**self.default_factory_kwargs))
        with pytest.raises(TypeError, match=r"Passed a perturber instance, expected type"):
            impl_class(perturber=FakePerturber(), **self.default_factory_kwargs)  # type: ignore[arg-type]

    # ===================== perturber_kwargs Parameter =====================

    def test_perturber_kwargs_passed_to_perturber(self) -> None:
        """perturber_kwargs are passed to created perturbers."""
        factory = self._make_factory(
            **(self.default_factory_kwargs | {"perturber_kwargs": {"param2": 99}}),
        )
        perturber = factory[0]
        config = perturber.get_config()
        assert config["param1"] == factory.thetas[0]
        assert config["param2"] == 99

    def test_theta_values_override_perturber_kwargs(self) -> None:
        """Theta values override perturber_kwargs for same key."""
        factory = self._make_factory(
            **(self.default_factory_kwargs | {"perturber_kwargs": {"param1": 999}}),  # should be overridden
        )
        perturber = factory[0]
        config = perturber.get_config()
        assert config["param1"] != 999
        assert config["param1"] == factory.thetas[0]  # theta value, not perturber_kwargs

    # =========================== Property Tests ===========================

    def test_len_matches_thetas_length(self) -> None:
        """len(factory) equals len(factory.thetas)."""
        factory = self._make_factory(**self.default_factory_kwargs)
        assert len(factory) == len(factory.thetas)

    def test_getitem_matches_iteration(self) -> None:
        """factory[i] matches i-th item from iteration.

        Verifies that indexing and iteration produce the same perturbers
        with identical configurations.
        """
        factory = self._make_factory(**self.default_factory_kwargs)
        iterated = list(factory)
        for i, perturber in enumerate(iterated):
            indexed = factory[i]
            assert perturber.get_config() == indexed.get_config()
