"""Test import guard behavior for MAITE datasets module."""

import pytest

from tests._utils.import_guard_tests_mixin import ImportGuardTestsMixin


@pytest.mark.maite
class TestDatasetsMAITEImportGuard(ImportGuardTestsMixin):
    MODULE_PATH = "nrtk.interop._maite.datasets"
    DEPS_TO_MOCK = ["maite"]
    CLASSES = [
        "MAITEImageClassificationDataset",
        "MAITEObjectDetectionDataset",
        "MAITEObjectDetectionTarget",
    ]
    ERROR_MATCH = (
        r"{class_name} requires the `maite` extra\. "
        r"Install with: `pip install nrtk\[maite\]`"
    )


@pytest.mark.maite
class TestDatasetsToolsImportGuard(ImportGuardTestsMixin):
    MODULE_PATH = "nrtk.interop._maite.datasets"
    DEPS_TO_MOCK = ["kwcoco"]
    CLASSES = [
        "COCOMAITEObjectDetectionDataset",
        "dataset_to_coco",
    ]
    ERROR_MATCH = (
        r"{class_name} requires the `maite` and `tools` extras\. "
        r"Install with: `pip install nrtk\[maite,tools\]`"
    )


@pytest.mark.maite
def test_datasets_maite_public_imports() -> None:
    """Canary test: FAIL if maite marker is used but maite classes can't be imported."""
    try:
        from nrtk.interop._maite.datasets import (
            MAITEImageClassificationDataset,
            MAITEObjectDetectionDataset,
            MAITEObjectDetectionTarget,
        )

        del MAITEImageClassificationDataset
        del MAITEObjectDetectionDataset
        del MAITEObjectDetectionTarget
    except ImportError as e:
        pytest.fail(
            f"Running with maite marker but classes not importable: {e}. Ensure maite extra is installed.",
        )


@pytest.mark.maite
@pytest.mark.tools
def test_datasets_tools_public_imports() -> None:
    """Canary test: FAIL if tools marker is used but tools classes can't be imported."""
    try:
        from nrtk.interop._maite.datasets import (
            COCOMAITEObjectDetectionDataset,
            dataset_to_coco,
        )

        del COCOMAITEObjectDetectionDataset
        del dataset_to_coco
    except ImportError as e:
        pytest.fail(
            f"Running with tools marker but classes not importable: {e}. Ensure maite and tools extras are installed.",
        )
