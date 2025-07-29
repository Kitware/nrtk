from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock


def is_available(module_name: str) -> bool:
    """Determines whether an optional dependency is available for use.

    Args:
        module_name (str): Name of the module to check.

    Returns:
        bool: True if the module is available, False if the module is
            not installed.
    """
    try:
        loader = importlib.util.find_spec(module_name)
        available = loader is not None
    except ValueError:
        available = False
    except ModuleNotFoundError:
        available = False

    return available


def _fake_spec(module_name: str) -> None:
    """Mocks out the __spec__ object for a module.

    Args:
        module_name (str): Name of the module to add a __spec__ attribute to.
    """
    from importlib.machinery import ModuleSpec

    sys.modules[module_name].__spec__ = ModuleSpec(name=module_name, loader=None, origin=None)


def _expand_submodules(submodules: Sequence[str] | None) -> Sequence[str]:
    """Expands a list of submodules to include all parent submodules individually.

       Useful because `import A.B.C` requires mocking out
       `A`, `A.B` and `A.B.C` individually.

    Example:
           submodules = ["A.B.C", "D"]
           returns: ["A", "A.B", "A.B.C", "D"]

    Args:
        submodules (Sequence[str] | None): Optional list of submodules to expand.

    Returns:
        Sequence[str]: Expanded list of submodules.
    """
    expanded = []
    if submodules:
        for submodule in submodules:
            parts = submodule.split(".")
            expanded.extend([".".join(parts[: i + 1]) for i in range(len(parts))])
    return expanded


def import_guard(  # noqa: C901
    module_name: str,
    exception: type[ImportError],
    submodules: Sequence[str] | None = None,
    objects: Sequence[str] | None = None,
    fake_spec: bool = False,
) -> bool:
    """Guards import statements for optionally installed dependencies.

    Args:
        module_name (str): Name of the module to import guard.
        exception (ImportError): Exception to raise if the module is
            not installed and code attempts to use it.
        submodules (Sequence[str] | None): An optional list of submodules
            to import guard. This function will be called recursively on
            all submodules and their parents. Ex:
                import_guard("example", ImportError, ["A.B.C"])
                will guard the following import statements:
                    import example
                    import example.A
                    import example.A.B
                    import example.A.B.C
        objects (Sequence[str] | None): An optional list of module
            attributes that should be treated as generic `object` type when
            the module is not found. Useful for preventing pyright errors
            when defining a class which inherits a type from an optional
            dependency.
        fake_spec (bool): A boolean flag that can be set to enable creation
            of a mocked __spec__ attribute on the imported module when the
            dependency is not found. Useful when making importlib calls related
            to the module that would fail when __spec__ is None.

    Returns:
        bool: True if the module is available, False if the module is
            unavailable and will be mocked out.
    """
    submodules = _expand_submodules(submodules)
    available = is_available(module_name)

    if not available:

        class ModuleUnavailable(MagicMock):
            def __call__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
                raise exception

            def is_usable(self) -> bool:
                return False

        sys.modules[module_name] = ModuleUnavailable()

    if submodules:
        for submodule in submodules:
            submodule_name = module_name + "." + submodule
            available = import_guard(submodule_name, exception, None, objects) and available

    if not available and objects:
        for obj in objects:
            setattr(sys.modules[module_name], obj, object)

    if not available and fake_spec:
        _fake_spec(module_name)

    return available
