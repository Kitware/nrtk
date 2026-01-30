"""Mixin providing shared tests for PerturbImage implementations.

This module provides ``PerturberTestsMixin``, a mixin class that defines shared
test cases for all PerturbImage implementations. Concrete perturber test classes
(e.g., TestAverageBlurPerturber, TestGaussianBlurPerturber) inherit from this
mixin to automatically run common tests.

Usage:
    Subclasses must define:
        - ``impl_class``: The PerturbImage implementation class being tested

Shared Test Cases:
    Plugin Discovery
        - Perturber is discoverable via PerturbImage.get_impls()
"""

from typing import ClassVar

from nrtk.interfaces.perturb_image import PerturbImage


class PerturberTestsMixin:
    """Mixin providing shared tests for PerturbImage implementations.

    See module docstring for full list of shared test cases and usage instructions.

    Attributes:
        impl_class: The PerturbImage implementation class being tested.
    """

    impl_class: ClassVar[type[PerturbImage]]

    # ========================== Plugin Discovery ==========================

    def test_plugin_discovery(self) -> None:
        """Perturber is discoverable via PerturbImage.get_impls()."""
        assert self.impl_class in PerturbImage.get_impls()
