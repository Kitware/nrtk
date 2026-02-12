"""Test import guard behavior for entrypoints module."""

import pytest

from tests._utils.import_guard_tests_mixin import ImportGuardTestsMixin


@pytest.mark.maite
class TestEntrypointsMaiteImportGuard(ImportGuardTestsMixin):
    """Test that nrtk_perturber raises ImportError when maite is missing."""

    MODULE_PATH = "nrtk.entrypoints"
    DEPS_TO_MOCK = ["maite"]
    CLASSES = ["nrtk_perturber", "nrtk_perturber_cli"]
    ERROR_MATCH = r"{class_name} requires the `maite`"


@pytest.mark.maite
class TestEntrypointsToolsImportGuard(ImportGuardTestsMixin):
    """Test that nrtk_perturber_cli raises ImportError when kwcoco is missing."""

    MODULE_PATH = "nrtk.entrypoints"
    DEPS_TO_MOCK = ["kwcoco"]
    CLASSES = ["nrtk_perturber_cli"]
    ALWAYS_AVAILABLE = ["nrtk_perturber"]
    ERROR_MATCH = (
        r"{class_name} requires the `maite` and `tools` extras\. "
        r"Install with: `pip install nrtk\[maite,tools\]`"
    )


@pytest.mark.maite
def test_maite_entrypoint_imports() -> None:
    """Canary test: FAIL if marker is used but entrypoint can't be imported."""
    try:
        from nrtk.entrypoints import nrtk_perturber

        del nrtk_perturber
    except ImportError as e:
        pytest.fail(
            f"Running with maite marker but nrtk_perturber not importable: {e}. Ensure maite extra is installed.",
        )


@pytest.mark.maite
@pytest.mark.tools
def test_tools_entrypoint_imports() -> None:
    """Canary test: FAIL if marker is used but CLI entrypoint can't be imported."""
    try:
        from nrtk.entrypoints import nrtk_perturber_cli

        del nrtk_perturber_cli
    except ImportError as e:
        pytest.fail(
            f"Running with tools marker but nrtk_perturber_cli not importable: {e}. "
            f"Ensure maite and tools extras are installed.",
        )
