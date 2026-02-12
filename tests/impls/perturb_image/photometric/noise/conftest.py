"""Pytest configuration for noise perturber tests.

Skips this directory if skimage extra is not installed.
"""


def pytest_ignore_collect() -> bool | None:
    """Skip this directory if noise perturbers are not importable."""
    try:
        from nrtk.impls.perturb_image.photometric.noise import (
            GaussianNoisePerturber,
            PepperNoisePerturber,
            SaltAndPepperNoisePerturber,
            SaltNoisePerturber,
            SpeckleNoisePerturber,
        )

        del (
            GaussianNoisePerturber,
            PepperNoisePerturber,
            SaltAndPepperNoisePerturber,
            SaltNoisePerturber,
            SpeckleNoisePerturber,
        )
    except ImportError:
        return True
    return None
