"""
This module defines the `LinSpacePerturbImageFactory` class, an implementation
of the `PerturbImageFactory` interface that generates `PerturbImage` instances
with a parameter (`theta_key`) varying over a specified range, using linearly spaced values.

Classes:
    LinSpacePerturbImageFactory: A factory class for creating `PerturbImage` instances
    where a specified parameter varies over a defined range in linearly spaced steps.

Dependencies:
    - numpy for generating linearly spaced values.
    - nrtk.interfaces for the `PerturbImage` and `PerturbImageFactory` interfaces.

Usage:
    To use `LinSpacePerturbImageFactory`, initialize it with a `PerturbImage` type, a `theta_key`
    to vary, and specify the start, stop, and number of steps. This factory can then be used to
    generate perturbed image instances with linearly spaced parameter variations.

Example:
    factory = LinSpacePerturbImageFactory(
        perturber=SomePerturbImageClass,
        theta_key="parameter",
        start=0.0,
        stop=10.0,
        step=5
    )
    theta_values = factory.thetas  # Access generated linearly spaced values
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from typing_extensions import override

from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory


class LinSpacePerturbImageFactory(PerturbImageFactory):
    """Simple PerturbImageFactory implementation to step through the given range of values."""

    def __init__(
        self,
        perturber: type[PerturbImage],
        theta_key: str,
        start: float,
        stop: float,
        step: int = 1,
    ) -> None:
        """Initialize the factory to produce PerturbImage instances of the given type.

        Initialize the factory to produce PerturbImage instances of the given type,
        varying the given ``theta_key`` parameter from start to stop with given step.

        :param perturber: Python implementation type of the PerturbImage interface
            to produce.

        :param theta_key: Perturber parameter to vary between instances.

        :param start: Initial value of desired range (inclusive).

        :param stop: Final value of desired range (exclusive).

        :param step: Number of instances to generate.

        :raises TypeError: Given a perturber instance instead of type.
        """
        super().__init__(perturber=perturber, theta_key=theta_key)

        self.start = start
        self.stop = stop
        self.step = step

    @override
    @property
    def thetas(self) -> Sequence[float]:
        if self.start == self.stop:
            return []
        return np.linspace(self.start, self.stop, self.step, endpoint=False).tolist()

    @override
    def get_config(self) -> dict[str, Any]:
        cfg = super().get_config()
        cfg["start"] = self.start
        cfg["stop"] = self.stop
        cfg["step"] = self.step
        return cfg
