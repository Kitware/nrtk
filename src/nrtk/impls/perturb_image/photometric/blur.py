"""Blur perturbers using cv2."""

_CV2_CLASSES = ["AverageBlurPerturber", "GaussianBlurPerturber", "MedianBlurPerturber"]

__all__: list[str] = list()

try:
    from nrtk.impls.perturb_image.photometric._blur.average_blur_perturber import (
        AverageBlurPerturber as AverageBlurPerturber,
    )
    from nrtk.impls.perturb_image.photometric._blur.gaussian_blur_perturber import (
        GaussianBlurPerturber as GaussianBlurPerturber,
    )
    from nrtk.impls.perturb_image.photometric._blur.median_blur_perturber import (
        MedianBlurPerturber as MedianBlurPerturber,
    )

    # Override __module__ to reflect the public API path for plugin discovery
    AverageBlurPerturber.__module__ = __name__
    GaussianBlurPerturber.__module__ = __name__
    MedianBlurPerturber.__module__ = __name__

    __all__ += _CV2_CLASSES
except ImportError:
    pass


def __getattr__(name: str) -> None:
    if name in _CV2_CLASSES:
        raise ImportError(
            f"{name} requires the `graphics` or `headless` extra. "
            f"Install with: `pip install nrtk[graphics]` or `pip install nrtk[headless]`",
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
