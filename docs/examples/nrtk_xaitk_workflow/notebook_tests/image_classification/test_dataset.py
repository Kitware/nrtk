"""Tests for image classification dataset import guards."""

import pytest

from tests._utils.import_guard_tests_mixin import ImportGuardTestsMixin


class TestICDatasetImportGuard(ImportGuardTestsMixin):
    """Test import guard for IC dataset helpers when dependencies are unavailable."""

    MODULE_PATH = "image_classification.dataset"
    DEPS_TO_MOCK = ["datasets", "maite", "PIL"]
    CLASSES = ["HuggingFaceMaiteDataset", "create_data_subset"]
    ERROR_MATCH = r"{class_name} requires additional dependencies"


@pytest.mark.xaitk
def test_ic_dataset_public_imports() -> None:
    """Canary test: FAIL if xaitk marker is used but IC dataset helpers can't be imported.

    When running ``pytest -m xaitk``, this test asserts that the environment was
    built with the required dependencies. If they are not installed, this test
    FAILS (not skips) to indicate a CI/environment configuration error.
    """
    try:
        from image_classification.dataset import (
            HuggingFaceMaiteDataset,
            create_data_subset,
        )

        del HuggingFaceMaiteDataset, create_data_subset
    except ImportError as e:
        pytest.fail(
            f"Running with xaitk marker but IC dataset helpers not importable: {e}. "
            f"Ensure dependencies are installed: pip install nrtk[maite,pillow] datasets",
        )
