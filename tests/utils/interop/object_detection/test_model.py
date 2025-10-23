import pytest

from nrtk.utils._exceptions import NRTKXAITKHelperImportError
from nrtk.utils.interop.object_detection.model import nrtk_xaitk_helpers_available


@pytest.mark.skipif(not nrtk_xaitk_helpers_available, reason=str(NRTKXAITKHelperImportError()))
def test_yolo_maite_model() -> None:
    pass
