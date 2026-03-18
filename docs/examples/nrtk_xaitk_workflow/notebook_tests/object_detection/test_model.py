"""Tests for object detection model import guards."""

import pytest

from tests._utils.import_guard_tests_mixin import ImportGuardTestsMixin


class TestODModelImportGuard(ImportGuardTestsMixin):
    """Test import guard for OD model helpers when dependencies are unavailable."""

    MODULE_PATH = "object_detection.model"
    DEPS_TO_MOCK = ["torch", "ultralytics", "maite"]
    CLASSES = ["MaiteYOLODetector"]
    ERROR_MATCH = r"{class_name} requires additional dependencies"
    ADDITIONAL_MODULES = ["object_detection.dataset"]


@pytest.mark.xaitk
def test_od_model_public_imports() -> None:
    """Canary test: FAIL if xaitk marker is used but OD model helpers can't be imported.

    When running ``pytest -m xaitk``, this test asserts that the environment was
    built with the required dependencies. If they are not installed, this test
    FAILS (not skips) to indicate a CI/environment configuration error.
    """
    try:
        from object_detection.model import MaiteYOLODetector

        del MaiteYOLODetector
    except ImportError as e:
        pytest.fail(
            f"Running with xaitk marker but OD model helpers not importable: {e}. "
            f"Ensure dependencies are installed: pip install nrtk[maite] torch ultralytics",
        )
