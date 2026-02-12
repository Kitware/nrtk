"""Random noise perturbers using skimage."""

_SKIMAGE_CLASSES = [
    "GaussianNoisePerturber",
    "PepperNoisePerturber",
    "SaltAndPepperNoisePerturber",
    "SaltNoisePerturber",
    "SpeckleNoisePerturber",
]

__all__: list[str] = []

try:
    from nrtk.impls.perturb_image.photometric._noise.gaussian_noise_perturber import (
        GaussianNoisePerturber as GaussianNoisePerturber,
    )
    from nrtk.impls.perturb_image.photometric._noise.pepper_noise_perturber import (
        PepperNoisePerturber as PepperNoisePerturber,
    )
    from nrtk.impls.perturb_image.photometric._noise.salt_and_pepper_noise_perturber import (
        SaltAndPepperNoisePerturber as SaltAndPepperNoisePerturber,
    )
    from nrtk.impls.perturb_image.photometric._noise.salt_noise_perturber import (
        SaltNoisePerturber as SaltNoisePerturber,
    )
    from nrtk.impls.perturb_image.photometric._noise.speckle_noise_perturber import (
        SpeckleNoisePerturber as SpeckleNoisePerturber,
    )

    # Override __module__ to reflect the public API path for plugin discovery
    GaussianNoisePerturber.__module__ = __name__
    PepperNoisePerturber.__module__ = __name__
    SaltAndPepperNoisePerturber.__module__ = __name__
    SaltNoisePerturber.__module__ = __name__
    SpeckleNoisePerturber.__module__ = __name__

    __all__ += _SKIMAGE_CLASSES
except ImportError:
    pass


def __getattr__(name: str) -> None:
    if name in _SKIMAGE_CLASSES:
        raise ImportError(
            f"{name} requires the `skimage` extra. Install with: `pip install nrtk[skimage]`",
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
