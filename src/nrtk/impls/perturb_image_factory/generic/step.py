"""
This module defines the `StepPerturbImageFactory` class, which is an implementation
of the `PerturbImageFactory` interface. The `StepPerturbImageFactory` class is designed
to generate a series of `PerturbImage` instances with a parameter (`theta_key`) that
steps through a specified range of values, enabling controlled variation in the
perturbation process.

Classes:
    StepPerturbImageFactory: Factory for producing `PerturbImage` instances with a specific
    parameter (`theta_key`) varying over a specified range.

Dependencies:
    - math for range calculations.
    - nrtk.interfaces for the `PerturbImage` and `PerturbImageFactory` interfaces.

Usage:
    Instantiate `StepPerturbImageFactory` with a `PerturbImage` type, a `theta_key` to vary,
    and the start, stop, and step values for the parameter range. This factory can then be
    used to generate perturbed image instances with controlled variations.

Example:
    factory = StepPerturbImageFactory(
        perturber=SomePerturbImageClass,
        theta_key="parameter",
        start=0.0,
        stop=10.0,
        step=1.0,
        to_int=False
    )
    theta_values = factory.thetas  # Access generated range of parameter values
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from typing_extensions import override

from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory


class StepPerturbImageFactory(PerturbImageFactory):
    """Simple PerturbImageFactory implementation to step through the given range of values."""

    def __init__(
        self,
        perturber: type[PerturbImage],
        theta_key: str,
        start: float,
        stop: float,
        step: float = 1.0,
        to_int: bool = True,
    ) -> None:
        """Initialize the factory to produce PerturbImage instances of the given type.

        Initialize the factory to produce PerturbImage instances of the given type,
        varying the given ``theta_key`` parameter from start to stop with given step.

        :param perturber: Python implementation type of the PerturbImage interface
            to produce.

        :param theta_key: Perturber parameter to vary between instances.

        :param start: Initial value of desired range (inclusive).

        :param stop: Final value of desired range (exclusive).

        :param step: Step value between instances.

        :param to_int: Boolean variable determining whether the thetas are cast as
                       ints or floats.

        :raises TypeError: Given a perturber instance instead of type.
        """
        super().__init__(perturber=perturber, theta_key=theta_key)

        self.to_int = to_int
        self.start = start
        self.stop = stop
        self.step = step

    @override
    @property
    def thetas(self) -> Sequence[float] | Sequence[int]:
        if not self.to_int:
            return [self.start + i * self.step for i in range(math.ceil((self.stop - self.start) / self.step))]
        return [int(self.start + i * self.step) for i in range(math.ceil((self.stop - self.start) / self.step))]

    @override
    def get_config(self) -> dict[str, Any]:
        cfg = super().get_config()
        cfg["start"] = self.start
        cfg["stop"] = self.stop
        cfg["step"] = self.step
        cfg["to_int"] = self.to_int
        return cfg
