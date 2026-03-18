"""Tests for object detection dataset import guards."""

import pytest

from tests._utils.import_guard_tests_mixin import ImportGuardTestsMixin


class TestODDatasetImportGuard(ImportGuardTestsMixin):
    """Test import guard for OD dataset helpers when dependencies are unavailable."""

    MODULE_PATH = "object_detection.dataset"
    DEPS_TO_MOCK = ["torch", "maite", "PIL"]
    CLASSES = ["VisDroneObjectDetectionDataset", "stratified_sample_dataset", "YOLODetectionTarget"]
    ERROR_MATCH = r"{class_name} requires additional dependencies"


@pytest.mark.xaitk
def test_od_dataset_public_imports() -> None:
    """Canary test: FAIL if xaitk marker is used but OD dataset helpers can't be imported.

    When running ``pytest -m xaitk``, this test asserts that the environment was
    built with the required dependencies. If they are not installed, this test
    FAILS (not skips) to indicate a CI/environment configuration error.
    """
    try:
        from object_detection.dataset import (
            VisDroneObjectDetectionDataset,
            YOLODetectionTarget,
            stratified_sample_dataset,
        )

        del VisDroneObjectDetectionDataset, YOLODetectionTarget, stratified_sample_dataset
    except ImportError as e:
        pytest.fail(
            f"Running with xaitk marker but OD dataset helpers not importable: {e}. "
            f"Ensure dependencies are installed: pip install nrtk[maite,pillow] torch",
        )
