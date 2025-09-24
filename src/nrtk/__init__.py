"""Define the nrtk package."""

import importlib.metadata

__version__ = importlib.metadata.version(__name__)


__all__ = ["__version__", "impls", "interfaces", "interop", "utils"]

from . import impls, interfaces, interop, utils
