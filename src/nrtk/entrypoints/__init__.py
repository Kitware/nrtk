"""Command-line interface tools."""

_MAITE_CLASSES = ["nrtk_perturber"]
_MAITE_TOOLS_CLASSES = ["nrtk_perturber_cli"]

__all__: list[str] = list()

try:
    from nrtk.entrypoints._nrtk_perturber import nrtk_perturber as nrtk_perturber

    __all__ += _MAITE_CLASSES
except ImportError:
    pass

try:
    from nrtk.entrypoints._nrtk_perturber_cli import nrtk_perturber_cli as nrtk_perturber_cli

    __all__ += _MAITE_TOOLS_CLASSES
except ImportError:
    pass


def __getattr__(name: str) -> None:
    if name in _MAITE_CLASSES:
        raise ImportError(
            f"{name} requires the `maite` extra. Install with: `pip install nrtk[maite]`",
        )
    if name in _MAITE_TOOLS_CLASSES:
        raise ImportError(
            f"{name} requires the `maite` and `tools` extras. Install with: `pip install nrtk[maite,tools]`",
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
