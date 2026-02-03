"""Test import guard behavior for diffusion perturbers."""

import pytest

from tests._utils.import_guard_tests_mixin import ImportGuardTestsMixin


class TestDiffusionImportGuard(ImportGuardTestsMixin):
    """Test import guard for diffusion perturbers when deps are unavailable."""

    MODULE_PATH = "nrtk.impls.perturb_image.generative"
    DEPS_TO_MOCK = ["torch", "diffusers", "PIL", "transformers"]
    CLASSES = ["DiffusionPerturber"]
    ERROR_MATCH = (
        r"{class_name} requires the `diffusion` extra\. "
        r"Install with: `pip install nrtk\[diffusion\]`"
    )


@pytest.mark.diffusion
def test_diffusion_public_imports() -> None:
    """Canary test: FAIL if diffusion marker is used but classes can't be imported.

    When running `pytest -m diffusion`, this test asserts that the environment was
    built with the diffusion extra. If the dependencies are not installed, this test
    FAILS (not skips) to indicate a CI/environment configuration error.
    """
    try:
        from nrtk.impls.perturb_image.generative import DiffusionPerturber

        del DiffusionPerturber
    except ImportError as e:
        pytest.fail(
            f"Running with diffusion marker but perturbers not importable: {e}. Ensure diffusion extra is installed.",
        )
