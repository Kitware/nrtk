"""Pytest configuration for diffusion perturber tests.

Skips this directory if diffusion extra is not installed.
"""


def pytest_ignore_collect() -> bool | None:
    """Skip this directory if diffusion perturbers are not importable."""
    try:
        from nrtk.impls.perturb_image.generative import DiffusionPerturber

        del DiffusionPerturber
    except ImportError:
        return True
    return None
