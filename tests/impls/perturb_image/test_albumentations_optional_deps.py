"""Test import guard behavior for albumentations perturber."""

import pytest

from tests._utils.import_guard_tests_mixin import ImportGuardTestsMixin

MODULE_PATH = "nrtk.impls.perturb_image"


class TestAlbumentationsImportGuard(ImportGuardTestsMixin):
    """Test import guard for albumentations perturber when albumentations is unavailable."""

    MODULE_PATH = MODULE_PATH
    DEPS_TO_MOCK = ["albumentations", "cv2"]
    CLASSES = ["AlbumentationsPerturber"]
    ERROR_MATCH = (
        r"{class_name} requires the `albumentations` and \(`graphics` or `headless`\) extras\. "
        r"Install with: `pip install nrtk\[albumentations,graphics\]` or `pip install nrtk\[albumentations,headless\]`"
    )


@pytest.mark.opencv
@pytest.mark.albumentations
def test_albumentations_public_imports() -> None:
    """Canary test: FAIL if markers are used but AlbumentationsPerturber can't be imported.

    When running `pytest -m albumentations`, this test asserts that the environment
    was built with both the albumentations and graphics/headless extras. If either
    is missing, this test FAILS (not skips) to indicate a CI/environment configuration error.
    """
    try:
        from nrtk.impls.perturb_image import AlbumentationsPerturber

        del AlbumentationsPerturber
    except ImportError as e:
        pytest.fail(
            f"Running with albumentations marker but AlbumentationsPerturber not importable: {e}. "
            f"Ensure albumentations and graphics/headless extras are installed.",
        )
