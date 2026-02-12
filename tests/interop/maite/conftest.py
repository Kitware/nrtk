"""Pytest configuration for MAITE interop tests."""

import importlib
from pathlib import Path

# Mapping of test filenames to the extra modules they require beyond maite
_FILE_DEPENDENCIES: dict[str, list[str]] = {
    "test_converters.py": ["kwcoco"],
    "test_app.py": ["kwcoco", "httpx"],
    "test_aukus_app.py": ["kwcoco", "httpx"],
    "test_coco_maite_object_detection_dataset.py": ["kwcoco"],
}


def _can_import(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    return True


def pytest_ignore_collect(collection_path: Path) -> bool | None:
    """Skip MAITE tests when dependencies are not installed."""
    if not _can_import("maite"):
        return True

    for dep in _FILE_DEPENDENCIES.get(collection_path.name, []):
        if not _can_import(dep):
            return True

    return None
