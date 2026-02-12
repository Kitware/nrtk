"""Test import guard behavior for MAITE metadata module."""

import pytest

from tests._utils.import_guard_tests_mixin import ImportGuardTestsMixin


@pytest.mark.maite
class TestMetadataImportGuard(ImportGuardTestsMixin):
    MODULE_PATH = "nrtk.interop._maite.metadata"
    DEPS_TO_MOCK = ["maite"]
    CLASSES = [
        "NRTKDatumMetadata",
    ]
    ERROR_MATCH = (
        r"{class_name} requires the `maite` extra\. "
        r"Install with: `pip install nrtk\[maite\]`"
    )


@pytest.mark.maite
def test_metadata_public_imports() -> None:
    """Canary test: FAIL if maite marker is used but classes can't be imported."""
    try:
        from nrtk.interop._maite.metadata import NRTKDatumMetadata

        del NRTKDatumMetadata
    except ImportError as e:
        pytest.fail(
            f"Running with maite marker but classes not importable: {e}. Ensure maite extra is installed.",
        )
