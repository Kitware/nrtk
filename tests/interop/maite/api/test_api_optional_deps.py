"""Test import guard behavior for MAITE API module."""

import pytest

from tests._utils.import_guard_tests_mixin import ImportGuardTestsMixin


class TestAPIImportGuard(ImportGuardTestsMixin):
    MODULE_PATH = "nrtk.interop._maite.api"
    DEPS_TO_MOCK = ["fastapi"]
    CLASSES = ["handle_post", "handle_aukus_post"]
    ERROR_MATCH = (
        r"{class_name} requires the `maite` and `tools` extras?\. "
        r"Install with: `pip install nrtk\[maite,tools\]`"
    )


@pytest.mark.maite
@pytest.mark.tools
def test_api_public_imports() -> None:
    """Canary test: FAIL if marker is used but API handlers can't be imported."""
    try:
        from nrtk.interop._maite.api import handle_aukus_post, handle_post

        del handle_post, handle_aukus_post
    except ImportError as e:
        pytest.fail(
            f"Running with tools marker but API handlers not importable: {e}. "
            f"Ensure maite and tools extras are installed.",
        )
