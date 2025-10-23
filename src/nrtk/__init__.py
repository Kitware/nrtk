"""Define the nrtk package."""

from collections.abc import Callable
from importlib import metadata
from typing import Any

import lazy_loader as lazy

__getattr__: Callable[[str], Any]
__dir__: Callable[[], list[str]]
__all__: list[str]

__version__ = metadata.version("nrtk")

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["impls", "interfaces", "interop", "utils"],
)
