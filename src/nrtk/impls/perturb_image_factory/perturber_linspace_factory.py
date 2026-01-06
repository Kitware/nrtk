"""Defines PerturberLinspaceFactory to create PerturbImage instances with parameters linearly spaced over a range.

Classes:
    PerturberLinspaceFactory: A factory class for creating `PerturbImage` instances
    where a specified parameter varies over a defined range in linearly spaced steps.

Dependencies:
    - numpy for generating linearly spaced values.
    - nrtk.interfaces for the `PerturbImage` and `PerturbImageFactory` interfaces.

Usage:
    To use `PerturberLinspaceFactory`, initialize it with a `PerturbImage` type, a `theta_key`
    to vary, and specify the start, stop, and number of samples. This factory can then be used to
    generate perturbed image instances with linearly spaced parameter variations.

Example:
    >>> from nrtk.impls.perturb_image.photometric.enhance import BrightnessPerturber
    >>> factory = PerturberLinspaceFactory(
    ...     perturber=BrightnessPerturber, theta_key="factor", start=0.0, stop=1.0, num=5
    ... )
"""

from __future__ import annotations

__all__ = ["PerturberLinspaceFactory"]

from collections.abc import Sequence
from typing import Any

import numpy as np
from typing_extensions import override

from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory


class PerturberLinspaceFactory(PerturbImageFactory):
    """Simple PerturbImageFactory implementation to step through the given range of values.

    Attributes:
        perturber (type[PerturbImage]):
            perturber type to produce
        theta_key (str):
            peturber parameter to modify
        start (float):
            initial value of range (inclusive)
        end (float):
            end value of range (exclusive)
        num (int):
            number of values between start and end
        endpoint (bool):
                Decides if stop is included.
    """

    def __init__(
        self,
        *,
        perturber: type[PerturbImage],
        theta_key: str,
        start: float,
        stop: float,
        num: int = 1,
        endpoint: bool = True,
    ) -> None:
        """Initialize the factory to produce PerturbImage instances of the given type.

        Initialize the factory to produce PerturbImage instances of the given type,
        varying the given ``theta_key`` parameter from start to stop with given num.

        Args:
            perturber:
                Python implementation type of the PerturbImage interface to produce.
            theta_key:
                Perturber parameter to vary between instances.
            start:
                Initial value of desired range (inclusive).
            stop:
                Final value of desired range (inclusive).
            num:
                Number of instances to generate.
            endpoint:
                Decides if stop is included.

        Raises:
            TypeError: Given a perturber instance instead of type.
        """
        super().__init__(perturber=perturber, theta_key=theta_key)

        self.start = start
        self.stop = stop
        self.num = num
        self.endpoint = endpoint

    @property
    @override
    def thetas(self) -> Sequence[float]:
        """Use linspace to generate the desired range of values."""
        return np.linspace(self.start, self.stop, self.num, endpoint=self.endpoint).tolist()

    @override
    def get_config(self) -> dict[str, Any]:
        cfg = super().get_config()
        cfg["start"] = self.start
        cfg["stop"] = self.stop
        cfg["num"] = self.num
        cfg["endpoint"] = self.endpoint
        return cfg
