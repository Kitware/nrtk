import pytest

from nrtk.utils._exceptions import NRTKXAITKHelperImportError
from nrtk.utils.interop.image_classification.model import nrtk_xaitk_helpers_available


@pytest.mark.skipif(not nrtk_xaitk_helpers_available, reason=str(NRTKXAITKHelperImportError()))
def test_hugging_face_maite_model() -> None:
    pass
