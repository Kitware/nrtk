"""Pytest configuration for blur perturber tests.

Skips this directory if graphics or headless extra is not installed.
"""


def pytest_ignore_collect() -> bool | None:
    """Skip this directory if blur perturbers are not importable."""
    try:
        from nrtk.impls.perturb_image.photometric.blur import (
            AverageBlurPerturber,
            GaussianBlurPerturber,
            MedianBlurPerturber,
        )

        del AverageBlurPerturber, GaussianBlurPerturber, MedianBlurPerturber
    except ImportError:
        return True
    return None
