"""Pytest configuration for water droplet perturber tests.

Skips this directory if waterdroplet extra is not installed.
"""


def pytest_ignore_collect() -> bool | None:
    """Skip this directory if water droplet perturbers are not importable."""
    try:
        from nrtk.impls.perturb_image.environment import WaterDropletPerturber

        del WaterDropletPerturber
    except ImportError:
        return True
    return None
