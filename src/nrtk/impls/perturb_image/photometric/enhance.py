"""Enhancement perturbers using PIL."""

_PILLOW_CLASSES = ["BrightnessPerturber", "ColorPerturber", "ContrastPerturber", "SharpnessPerturber"]

try:
    from nrtk.impls.perturb_image.photometric._enhance.brightness_perturber import (
        BrightnessPerturber as BrightnessPerturber,
    )
    from nrtk.impls.perturb_image.photometric._enhance.color_perturber import (
        ColorPerturber as ColorPerturber,
    )
    from nrtk.impls.perturb_image.photometric._enhance.contrast_perturber import (
        ContrastPerturber as ContrastPerturber,
    )
    from nrtk.impls.perturb_image.photometric._enhance.sharpness_perturber import (
        SharpnessPerturber as SharpnessPerturber,
    )

    # Override __module__ to reflect the public API path for plugin discovery
    BrightnessPerturber.__module__ = __name__
    ColorPerturber.__module__ = __name__
    ContrastPerturber.__module__ = __name__
    SharpnessPerturber.__module__ = __name__

    __all__ = _PILLOW_CLASSES
except ImportError:
    __all__: list[str] = list()

    def __getattr__(name: str) -> None:
        if name in _PILLOW_CLASSES:
            raise ImportError(
                f"{name} requires the `pillow` extra. Install with: `pip install nrtk[pillow]`",
            )
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
