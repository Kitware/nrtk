"""Test import guard behavior for geometric perturbers."""

import pytest

from tests._utils.import_guard_tests_mixin import ImportGuardTestsMixin
from tests.impls.perturb_image.test_albumentations_optional_deps import (
    MODULE_PATH as ALBUMENTATIONS_MODULE_PATH,
)


class TestGeometricImportGuard(ImportGuardTestsMixin):
    """Test import guard for geometric perturbers when albumentations is unavailable."""

    MODULE_PATH = "nrtk.impls.perturb_image.geometric.random"
    DEPS_TO_MOCK = ["albumentations", "cv2"]
    CLASSES = ["RandomRotationPerturber", "RandomScalePerturber"]
    ALWAYS_AVAILABLE = ["RandomCropPerturber", "RandomTranslationPerturber"]
    ERROR_MATCH = (
        r"{class_name} requires the `albumentations` and \(`graphics` or `headless`\) extras\. "
        r"Install with: `pip install nrtk\[albumentations,graphics\]` or `pip install nrtk\[albumentations,headless\]`"
    )
    # The geometric perturbers import AlbumentationsPerturber from the wrapper module,
    # which has its own import guard. We must reset the wrapper module so that when
    # rotation.py and scale.py are re-imported, they see the mocked (unavailable) albumentations.
    ADDITIONAL_MODULES = [ALBUMENTATIONS_MODULE_PATH]


@pytest.mark.opencv
@pytest.mark.albumentations
def test_albumentations_public_imports() -> None:
    """Canary test: FAIL if markers are used but geometric perturbers can't be imported.

    When running `pytest -m albumentations`, this test asserts that the environment
    was built with both the albumentations and graphics/headless extras. If either
    is missing, this test FAILS (not skips) to indicate a CI/environment configuration error.
    """
    try:
        from nrtk.impls.perturb_image.geometric.random import (
            RandomRotationPerturber,
            RandomScalePerturber,
        )

        del RandomRotationPerturber, RandomScalePerturber
    except ImportError as e:
        pytest.fail(
            f"Running with albumentations marker but geometric perturbers not importable: {e}. "
            f"Ensure albumentations and graphics/headless extras are installed.",
        )
