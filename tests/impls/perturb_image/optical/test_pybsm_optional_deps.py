"""Test import guard behavior for pyBSM optical perturbers."""

import pytest

from tests._utils.import_guard_tests_mixin import ImportGuardTestsMixin


class TestPybsmPerturberImportGuard(ImportGuardTestsMixin):
    """Test import guard for PybsmPerturber when pybsm is unavailable."""

    MODULE_PATH = "nrtk.impls.perturb_image.optical"
    DEPS_TO_MOCK = ["pybsm"]
    CLASSES = [
        "PybsmPerturber",
    ]
    ERROR_MATCH = (
        r"{class_name} requires the `pybsm` extra\. "
        r"Install with: `pip install nrtk\[pybsm\]`"
    )


class TestOtfImportGuard(ImportGuardTestsMixin):
    """Test import guard for OTF perturbers when pybsm is unavailable."""

    MODULE_PATH = "nrtk.impls.perturb_image.optical.otf"
    DEPS_TO_MOCK = ["pybsm"]
    CLASSES = [
        "CircularAperturePerturber",
        "DefocusPerturber",
        "DetectorPerturber",
        "JitterPerturber",
        "TurbulenceAperturePerturber",
    ]
    ERROR_MATCH = (
        r"{class_name} requires the `pybsm` extra\. "
        r"Install with: `pip install nrtk\[pybsm\]`"
    )


@pytest.mark.pybsm
def test_pybsm_public_imports() -> None:
    """Canary test: FAIL if pybsm marker is used but perturbers can't be imported.

    When running `pytest -m pybsm`, this test asserts that the environment was
    built with the pybsm extra. If pybsm is not installed, this test
    FAILS (not skips) to indicate a CI/environment configuration error.
    """
    try:
        from nrtk.impls.perturb_image.optical import PybsmPerturber
        from nrtk.impls.perturb_image.optical.otf import (
            CircularAperturePerturber,
            DefocusPerturber,
            DetectorPerturber,
            JitterPerturber,
            TurbulenceAperturePerturber,
        )

        del (
            PybsmPerturber,
            CircularAperturePerturber,
            DefocusPerturber,
            DetectorPerturber,
            JitterPerturber,
            TurbulenceAperturePerturber,
        )
    except ImportError as e:
        pytest.fail(
            f"Running with pybsm marker but perturbers not importable: {e}. Ensure pybsm extra is installed.",
        )
