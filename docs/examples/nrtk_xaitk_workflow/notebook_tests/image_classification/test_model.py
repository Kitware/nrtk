"""Tests for image classification model import guards."""

import pytest

from tests._utils.import_guard_tests_mixin import ImportGuardTestsMixin


class TestICModelImportGuard(ImportGuardTestsMixin):
    """Test import guard for IC model helpers when dependencies are unavailable."""

    MODULE_PATH = "image_classification.model"
    DEPS_TO_MOCK = ["torch", "maite", "PIL", "transformers"]
    CLASSES = ["HuggingFaceMaiteModel"]
    ERROR_MATCH = r"{class_name} requires additional dependencies"


@pytest.mark.xaitk
def test_ic_model_public_imports() -> None:
    """Canary test: FAIL if xaitk marker is used but IC model helpers can't be imported.

    When running ``pytest -m xaitk``, this test asserts that the environment was
    built with the required dependencies. If they are not installed, this test
    FAILS (not skips) to indicate a CI/environment configuration error.
    """
    try:
        from image_classification.model import HuggingFaceMaiteModel

        del HuggingFaceMaiteModel
    except ImportError as e:
        pytest.fail(
            f"Running with xaitk marker but IC model helpers not importable: {e}. "
            f"Ensure dependencies are installed: pip install nrtk[maite,pillow] torch transformers",
        )
