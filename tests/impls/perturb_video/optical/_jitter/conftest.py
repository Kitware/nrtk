"""Pytest configuration for jitter video perturber tests.

Skips this directory if the jitter extra (scipy) is not installed. Gates on scipy directly so
a broken package import errors loudly instead of being silently skipped.
"""


def pytest_ignore_collect() -> bool | None:
    """Skip this directory if scipy (the jitter extra) is not installed."""
    try:
        import scipy

        del scipy
    except ImportError:
        return True
    return None
