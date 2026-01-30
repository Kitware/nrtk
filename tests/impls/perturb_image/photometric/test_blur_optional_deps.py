"""Test import guard behavior for blur perturbers."""

import pytest

from tests._utils.import_guard_tests_mixin import ImportGuardTestsMixin


class TestBlurImportGuard(ImportGuardTestsMixin):
    """Test import guard for blur perturbers when cv2 is unavailable."""

    MODULE_PATH = "nrtk.impls.perturb_image.photometric.blur"
    DEPS_TO_MOCK = ["cv2"]
    CLASSES = ["AverageBlurPerturber", "GaussianBlurPerturber", "MedianBlurPerturber"]
    ERROR_MATCH = (
        r"{class_name} requires the `graphics` or `headless` extra\. "
        r"Install with: `pip install nrtk\[graphics\]` or `pip install nrtk\[headless\]`"
    )


@pytest.mark.opencv
def test_opencv_public_imports() -> None:
    """Canary test: FAIL if opencv marker is used but blur perturbers can't be imported.

    When running `pytest -m opencv`, this test asserts that the environment was
    built with the graphics/headless extra. If cv2 is not installed, this test
    FAILS (not skips) to indicate a CI/environment configuration error.
    """
    try:
        from nrtk.impls.perturb_image.photometric.blur import (
            AverageBlurPerturber,
            GaussianBlurPerturber,
            MedianBlurPerturber,
        )

        del AverageBlurPerturber, GaussianBlurPerturber, MedianBlurPerturber
    except ImportError as e:
        pytest.fail(
            f"Running with opencv marker but blur perturbers not importable: {e}. "
            f"Ensure graphics or headless extra is installed.",
        )
