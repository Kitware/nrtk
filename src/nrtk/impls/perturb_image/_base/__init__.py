"""Private base classes for perturb_image implementations."""

from nrtk.impls.perturb_image._base._numpy_random_perturb_image import NumpyRandomPerturbImage

__all__ = ["NumpyRandomPerturbImage"]

_TORCH_CLASSES = ["TorchRandomPerturbImage"]

try:
    from nrtk.impls.perturb_image._base._torch_random_perturb_image import (
        TorchRandomPerturbImage as TorchRandomPerturbImage,
    )

    __all__ += _TORCH_CLASSES
except ImportError:
    pass


def __getattr__(name: str) -> None:
    if name in _TORCH_CLASSES:
        raise ImportError(
            f"{name} requires torch. Install with: `pip install nrtk[diffusion]`",
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
