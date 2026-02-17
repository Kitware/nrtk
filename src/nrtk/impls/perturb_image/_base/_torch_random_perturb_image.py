"""Private helper base class for perturbers using PyTorch's Generator.

Classes:
    TorchRandomPerturbImage: For perturbers using PyTorch's Generator.

Note:
    This is a private implementation detail. Use the public
    RandomPerturbImage interface instead.
"""

from __future__ import annotations

import warnings

__all__ = ["TorchRandomPerturbImage"]

from typing import Any

import torch
from typing_extensions import override

from nrtk.interfaces._random_perturb_image import RandomPerturbImage


class TorchRandomPerturbImage(RandomPerturbImage):
    """Base class for perturbers using PyTorch's Generator.

    Creates a torch Generator on the appropriate device, seeded with self._seed.
    The generator is stored as self._generator for use in pipeline calls.

    Note:
        Subclasses must set self._device before calling super().__init__().
    """

    _device: str
    _generator: Any

    @override
    def _set_seed(self) -> None:
        """Initialize torch Generator with seed."""
        if self._seed is None:
            self._generator = torch.Generator(device=self._get_device())
        else:
            self._generator = torch.Generator(device=self._get_device()).manual_seed(self._seed)

    def _get_device(self) -> str:
        """Get the device to use based on user preference or CUDA availability."""
        if self._device == "cuda" and not torch.cuda.is_available():
            warnings.warn(
                "CUDA is not available, but was requested. Falling back to CPU.",
                UserWarning,
                stacklevel=2,
            )
            return "cpu"
        return self._device
