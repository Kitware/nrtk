"""Tests for image classification model utilities."""

import sys
from pathlib import Path

import pytest

# Add utils to path
notebook_dir = Path(__file__).parent.parent.parent
utils_path = notebook_dir / "utils"
if str(utils_path) not in sys.path:
    sys.path.insert(0, str(utils_path))

from image_classification.model import nrtk_xaitk_helpers_available  # noqa: E402

from nrtk.utils._exceptions import NRTKXAITKHelperImportError  # noqa: E402


@pytest.mark.skipif(not nrtk_xaitk_helpers_available, reason=str(NRTKXAITKHelperImportError()))
def test_hugging_face_maite_model() -> None:
    """Test stub for HuggingFace MAITE model."""
    pass
