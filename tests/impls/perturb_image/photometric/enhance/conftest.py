"""Pytest configuration for enhancement perturber tests.

Skips this directory if pillow extra is not installed.
"""


def pytest_ignore_collect() -> bool | None:
    """Skip this directory if enhancement perturbers are not importable."""
    try:
        from nrtk.impls.perturb_image.photometric.enhance import (
            BrightnessPerturber,
            ColorPerturber,
            ContrastPerturber,
            SharpnessPerturber,
        )

        del BrightnessPerturber, ColorPerturber, ContrastPerturber, SharpnessPerturber
    except ImportError:
        return True
    return None
