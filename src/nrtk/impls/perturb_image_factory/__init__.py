"""Module for all implementations of PerturbImageFactory."""

from collections.abc import Callable
from typing import Any

import lazy_loader as lazy

__getattr__: Callable[[str], Any]
__dir__: Callable[[], list[str]]
__all__: list[str]

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "perturber_linspace_factory",
        "perturber_multivariate_factory",
        "perturber_one_step_factory",
        "perturber_step_factory",
    ],
)
