from collections.abc import Sequence
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import pytest
from maite.protocols import DatumMetadata

from nrtk.interop.maite.metadata.datum import NRTKDatumMetadata, _forward_md_keys
from nrtk.interop.maite.utils.detection import maite_available
from nrtk.utils._exceptions import MaiteImportError


@pytest.mark.skipif(not maite_available, reason=str(MaiteImportError()))
class TestGenericInteropUtilities:
    @pytest.mark.parametrize(
        ("md", "aug_md", "forwarded_keys", "expectation"),
        [
            ({"id": 1}, {"id": 1, "aug_md": "val1"}, ["id"], does_not_raise()),
            (
                {"id": 0, "not_forwarded": "val1", "collision": "val2"},
                {"id": 0, "collision": "val3", "aug_md": "test"},
                ["id"],
                pytest.raises(KeyError, match=r"already present in metadata"),
            ),
        ],
    )
    def test_forward_md_keys(
        self,
        md: DatumMetadata,  # pyright: ignore [reportInvalidTypeForm]
        aug_md: NRTKDatumMetadata,
        forwarded_keys: Sequence[str],
        expectation: AbstractContextManager,
    ) -> None:
        with expectation:
            md_out = _forward_md_keys(md, aug_md, forwarded_keys)

            for key in md_out:
                assert key in md or aug_md
