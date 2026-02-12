"""Pytest configuration for pyBSM perturber tests.

Skips this directory if pybsm extra is not installed.
"""


def pytest_ignore_collect() -> bool | None:
    """Skip this directory if pybsm perturbers are not importable."""
    try:
        from nrtk.impls.perturb_image.optical import PybsmPerturber
        from nrtk.impls.perturb_image.optical.otf import (
            CircularAperturePerturber,
            DefocusPerturber,
            DetectorPerturber,
            JitterPerturber,
            TurbulenceAperturePerturber,
        )

        del (
            PybsmPerturber,
            CircularAperturePerturber,
            DefocusPerturber,
            DetectorPerturber,
            JitterPerturber,
            TurbulenceAperturePerturber,
        )
    except ImportError:
        return True
    return None
