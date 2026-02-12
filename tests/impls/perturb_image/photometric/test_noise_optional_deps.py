"""Test import guard behavior for noise perturbers."""

import pytest

from tests._utils.import_guard_tests_mixin import ImportGuardTestsMixin


class TestNoiseImportGuard(ImportGuardTestsMixin):
    """Test import guard for noise perturbers when skimage is unavailable."""

    MODULE_PATH = "nrtk.impls.perturb_image.photometric.noise"
    DEPS_TO_MOCK = ["skimage.util"]
    CLASSES = [
        "GaussianNoisePerturber",
        "PepperNoisePerturber",
        "SaltAndPepperNoisePerturber",
        "SaltNoisePerturber",
        "SpeckleNoisePerturber",
    ]
    ERROR_MATCH = (
        r"{class_name} requires the `skimage` extra\. "
        r"Install with: `pip install nrtk\[skimage\]`"
    )


@pytest.mark.skimage
def test_skimage_public_imports() -> None:
    """Canary test: FAIL if skimage marker is used but noise perturbers can't be imported.

    When running `pytest -m skimage`, this test asserts that the environment was
    built with the skimage extra. If skimage is not installed, this test
    FAILS (not skips) to indicate a CI/environment configuration error.
    """
    try:
        from nrtk.impls.perturb_image.photometric.noise import (
            GaussianNoisePerturber,
            PepperNoisePerturber,
            SaltAndPepperNoisePerturber,
            SaltNoisePerturber,
            SpeckleNoisePerturber,
        )

        del (
            GaussianNoisePerturber,
            PepperNoisePerturber,
            SaltAndPepperNoisePerturber,
            SaltNoisePerturber,
            SpeckleNoisePerturber,
        )
    except ImportError as e:
        pytest.fail(
            f"Running with skimage marker but noise perturbers not importable: {e}. Ensure skimage extra is installed.",
        )
