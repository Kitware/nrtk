"""Test import guard behavior for enhancement perturbers."""

import pytest

from tests._utils.import_guard_tests_mixin import ImportGuardTestsMixin


class TestEnhanceImportGuard(ImportGuardTestsMixin):
    """Test import guard for enhancement perturbers when PIL is unavailable."""

    MODULE_PATH = "nrtk.impls.perturb_image.photometric.enhance"
    DEPS_TO_MOCK = ["PIL"]
    CLASSES = ["BrightnessPerturber", "ColorPerturber", "ContrastPerturber", "SharpnessPerturber"]
    ERROR_MATCH = (
        r"{class_name} requires the `pillow` extra\. "
        r"Install with: `pip install nrtk\[pillow\]`"
    )


@pytest.mark.pillow
def test_pillow_public_imports() -> None:
    """Canary test: FAIL if pillow marker is used but enhance perturbers can't be imported.

    When running `pytest -m pillow`, this test asserts that the environment was
    built with the pillow extra. If PIL is not installed, this test
    FAILS (not skips) to indicate a CI/environment configuration error.
    """
    try:
        from nrtk.impls.perturb_image.photometric.enhance import (
            BrightnessPerturber,
            ColorPerturber,
            ContrastPerturber,
            SharpnessPerturber,
        )

        del BrightnessPerturber, ColorPerturber, ContrastPerturber, SharpnessPerturber
    except ImportError as e:
        pytest.fail(
            f"Running with pillow marker but enhance perturbers not importable: {e}. Ensure pillow extra is installed.",
        )
