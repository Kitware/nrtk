"""Private helper base class for perturbers using numpy's random Generator.

Classes:
    NumpyRandomPerturbImage: For perturbers using numpy's random Generator.

Note:
    This is a private implementation detail. Use the public
    RandomPerturbImage interface instead.
"""

from __future__ import annotations

__all__ = ["NumpyRandomPerturbImage"]

import numpy as np
from typing_extensions import override

from nrtk.interfaces._random_perturb_image import RandomPerturbImage


class NumpyRandomPerturbImage(RandomPerturbImage):
    """Base class for perturbers using numpy's random Generator.

    Initializes self._rng as a numpy Generator seeded with self._seed.
    When seed is None, creates an unseeded Generator for non-deterministic behavior.
    """

    _rng: np.random.Generator

    @override
    def _set_seed(self) -> None:
        """Initialize numpy random Generator with seed."""
        self._rng = np.random.default_rng(self._seed)
