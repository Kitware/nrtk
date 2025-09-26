import pytest

from nrtk.utils._exceptions import NRTKXAITKHelperImportError
from nrtk.utils.interop.object_detection.dataset import nrtk_xaitk_helpers_available


@pytest.mark.skipif(not nrtk_xaitk_helpers_available, reason=str(NRTKXAITKHelperImportError()))
def test_visdrone_maite_dataset() -> None:
    pass
