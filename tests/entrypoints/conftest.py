"""Pytest configuration for maite bin tests.

Skips test_nrtk_perturber_opencv.py if graphics or headless extra is not installed.
"""

from pathlib import Path


def pytest_ignore_collect(collection_path: Path) -> bool | None:
    """Skip opencv-dependent test file if blur perturbers are not importable."""
    if collection_path.name == "test_nrtk_perturber_opencv.py":
        try:
            from nrtk.impls.perturb_image.photometric.blur import AverageBlurPerturber

            del AverageBlurPerturber
        except ImportError:
            return True
    return None
