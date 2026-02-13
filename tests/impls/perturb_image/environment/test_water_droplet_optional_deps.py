"""Test import guard behavior for water droplet perturbers."""

import pytest

from tests._utils.import_guard_tests_mixin import ImportGuardTestsMixin


class TestWaterDropletImportGuard(ImportGuardTestsMixin):
    """Test import guard for water droplet perturbers when scipy/numba are unavailable."""

    MODULE_PATH = "nrtk.impls.perturb_image.environment"
    ALWAYS_AVAILABLE = ["HazePerturber"]
    DEPS_TO_MOCK = ["numba", "scipy"]
    CLASSES = ["WaterDropletPerturber"]
    ERROR_MATCH = (
        r"{class_name} requires the `waterdroplet` extra\. "
        r"Install with: `pip install nrtk\[waterdroplet\]`"
    )


@pytest.mark.waterdroplet
def test_waterdroplet_public_imports() -> None:
    """Canary test: FAIL if waterdroplet marker is used but classes can't be imported.

    When running `pytest -m waterdroplet`, this test asserts that the environment was
    built with the waterdroplet extra. If scipy/numba are not installed, this test
    FAILS (not skips) to indicate a CI/environment configuration error.
    """
    try:
        from nrtk.impls.perturb_image.environment import WaterDropletPerturber

        del WaterDropletPerturber
    except ImportError as e:
        pytest.fail(
            f"Running with waterdroplet marker but perturbers not importable: {e}. "
            f"Ensure waterdroplet extra is installed.",
        )
