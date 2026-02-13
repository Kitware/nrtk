"""Pytest configuration for albumentations perturber tests.

Skips this directory if albumentations extra is not installed.
"""


def pytest_ignore_collect() -> bool | None:
    """Skip this directory if AlbumentationsPerturber is not importable."""
    try:
        from nrtk.impls.perturb_image import AlbumentationsPerturber

        del AlbumentationsPerturber
    except ImportError:
        return True
    return None
