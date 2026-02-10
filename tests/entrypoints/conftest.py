"""Pytest configuration for entrypoints tests."""

from pathlib import Path

# Files that require the tools extra (kwcoco, click) in addition to maite
_TOOLS_FILES = {
    "test_nrtk_perturber_cli.py",
}


def pytest_ignore_collect(collection_path: Path) -> bool | None:
    """Skip entrypoints tests when dependencies are not installed."""
    # All tests in this directory require maite
    try:
        from nrtk.entrypoints import nrtk_perturber

        del nrtk_perturber
    except ImportError:
        return True

    # Some test files also require the tools extra (kwcoco, click)
    if collection_path.name in _TOOLS_FILES:
        try:
            import kwcoco  # noqa: F401
        except ImportError:
            return True

    return None
