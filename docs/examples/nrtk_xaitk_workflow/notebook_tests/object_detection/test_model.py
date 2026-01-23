"""Tests for object detection model utilities."""

import pytest

from nrtk.utils._exceptions import NRTKXAITKHelperImportError
from nrtk.utils._import_guard import import_guard

PIL_available: bool = import_guard(module_name="PIL", exception=NRTKXAITKHelperImportError)
maite_available: bool = import_guard(
    module_name="maite",
    exception=NRTKXAITKHelperImportError,
    submodules=["protocols.image_classification"],
    objects=["Dataset", "DatumMetadataType", "InputType", "TargetType"],
)
datasets_available: bool = import_guard(module_name="datasets", exception=NRTKXAITKHelperImportError)
nrtk_xaitk_helpers_available: bool = maite_available and PIL_available and datasets_available


@pytest.mark.skipif(not nrtk_xaitk_helpers_available, reason=str(NRTKXAITKHelperImportError()))
def test_yolo_maite_model() -> None:
    """Test stub for YOLO MAITE detector."""
    pass
