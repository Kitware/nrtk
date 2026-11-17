"""Pytest configuration for HCIPy video perturber tests.

Skips this directory if the hcipy and scipy extras are not installed.
"""


def pytest_ignore_collect() -> bool | None:
    """Skip this directory if HCIPy perturbers are not importable."""
    try:
        # Opt in before probing: the guard checks the experimental gate before the
        # extras gate, so without this the probe raises the experimental error and
        # the directory silently skips even with hcipy installed.
        import nrtk.experimental  # noqa: F401
        from nrtk.impls.perturb_video.optical import TurbulenceVideoPerturber

        del TurbulenceVideoPerturber
    except ImportError:
        return True
    return None
