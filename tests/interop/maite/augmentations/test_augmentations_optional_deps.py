"""Test import guard behavior for MAITE augmentations module."""

import pytest

from tests._utils.import_guard_tests_mixin import ImportGuardTestsMixin


class TestAugmentationsImportGuard(ImportGuardTestsMixin):
    MODULE_PATH = "nrtk.interop._maite.augmentations"
    DEPS_TO_MOCK = ["maite"]
    CLASSES = [
        "MAITEImageClassificationAugmentation",
        "MAITEObjectDetectionAugmentation",
    ]
    ERROR_MATCH = (
        r"{class_name} requires the `maite` extra\. "
        r"Install with: `pip install nrtk\[maite\]`"
    )


@pytest.mark.maite
def test_augmentations_public_imports() -> None:
    """Canary test: FAIL if maite marker is used but classes can't be imported."""
    try:
        from nrtk.interop._maite.augmentations import (
            MAITEImageClassificationAugmentation,
            MAITEObjectDetectionAugmentation,
        )

        del MAITEImageClassificationAugmentation
        del MAITEObjectDetectionAugmentation
    except ImportError as e:
        pytest.fail(
            f"Running with maite marker but classes not importable: {e}. Ensure maite extra is installed.",
        )
