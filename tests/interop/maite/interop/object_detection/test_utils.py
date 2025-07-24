import json
import unittest.mock as mock
from collections.abc import Sequence
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import py  # type: ignore
import pytest

from nrtk.interop.maite.interop.object_detection.dataset import (
    COCOJATICObjectDetectionDataset,
    JATICDetectionTarget,
    JATICObjectDetectionDataset,
)
from nrtk.interop.maite.interop.object_detection.utils import dataset_to_coco, kwcoco_available, maite_available
from nrtk.utils._exceptions import KWCocoImportError, MaiteImportError

if kwcoco_available:
    # Multiple type ignores added for pyright's handling of guarded imports
    import kwcoco  # type: ignore


TargetType: type = object
DatumMetadataType: type = TypedDict
if maite_available:
    # Multiple type ignores added for pyright's handling of guarded imports
    from maite.protocols.object_detection import DatumMetadataType, TargetType

random = np.random.default_rng()


@pytest.mark.skipif(not kwcoco_available, reason=str(KWCocoImportError()))
@pytest.mark.skipif(not maite_available, reason=str(MaiteImportError()))
@pytest.mark.parametrize(
    ("imgs", "input_dets", "datum_metadata", "dataset_id", "img_filenames", "categories", "expectation"),
    [
        (
            [random.integers(0, 255, size=(3, 10, 10), dtype=np.uint8)],
            [
                JATICDetectionTarget(
                    boxes=random.integers(0, 4, size=(2, 4)),
                    labels=random.integers(0, 2, size=(2,)),
                    scores=random.random(2),
                ),
            ],
            [{"id": 0}],
            "dummy_dataset",
            ["images/img1.png"],
            [
                {"id": 0, "name": "cat0", "supercategory": None},
                {"id": 1, "name": "cat1", "supercategory": None},
            ],
            does_not_raise(),
        ),
        (
            [random.integers(0, 255, size=(3, 3, 3), dtype=np.uint8)] * 2,
            [
                JATICDetectionTarget(
                    boxes=random.integers(0, 4, size=(2, 4)),
                    labels=random.integers(0, 2, size=(2,)),
                    scores=random.random(2),
                ),
            ]
            * 2,
            [{"id": idx} for idx in range(2)],
            "dummy_dataset",
            ["images/img1.png"],
            [
                {"id": 0, "name": "cat0", "supercategory": None},
                {"id": 1, "name": "cat1", "supercategory": None},
            ],
            pytest.raises(ValueError, match=r"Image filename and dataset length mismatch"),
        ),
    ],
)
def test_dataset_to_coco(
    imgs: Sequence[np.ndarray],
    input_dets: Sequence[TargetType],  # pyright: ignore [reportInvalidTypeForm]
    datum_metadata: Sequence[DatumMetadataType],  # pyright: ignore [reportInvalidTypeForm]
    dataset_id: str,
    img_filenames: list[Path],
    categories: list[dict[str, Any]],
    expectation: AbstractContextManager,
    tmpdir: py.path.local,
) -> None:
    """Test that a MAITE dataset is appropriately saved to file in COCO format."""
    tmp_path = Path(tmpdir)

    dataset = JATICObjectDetectionDataset(
        imgs=imgs,
        dets=input_dets,
        datum_metadata=datum_metadata,
        dataset_id=dataset_id,
    )

    with expectation:
        dataset_to_coco(
            dataset=dataset,  # pyright: ignore [reportPossiblyUnboundVariable]
            output_dir=Path(tmpdir),
            img_filenames=img_filenames,
            dataset_categories=categories,
        )

        # Confirm annotations and metadata files exist
        label_file = tmp_path / "annotations.json"
        assert label_file.is_file()
        metadata_file = tmp_path / "image_metadata.json"
        assert metadata_file.is_file()

        # Confirm images exist
        img_paths = [tmp_path / filename for filename in img_filenames]
        for path in img_paths:
            assert path.is_file()

        with open(metadata_file) as f:
            metadata = json.load(f)

        # Re-create MAITE dataset from file
        coco_dataset = COCOJATICObjectDetectionDataset(
            kwcoco_dataset=kwcoco.CocoDataset(label_file),  # pyright: ignore [reportPossiblyUnboundVariable, reportCallIssue]
            image_metadata=metadata,
        )

        assert len(dataset) == len(coco_dataset)
        for i in range(len(dataset)):
            image, dets, md = dataset[i]
            c_image, c_dets, c_md = coco_dataset[i]

            assert np.array_equal(image, c_image)  # pyright: ignore [reportArgumentType]
            assert np.array_equal(dets.boxes, c_dets.boxes)  # pyright: ignore [reportAttributeAccessIssue]
            assert np.array_equal(dets.labels, c_dets.labels)  # pyright: ignore [reportAttributeAccessIssue]
            # Not checking scores as they are not perserved

            # Not checking for total equality because the COCO dataset class adds metadata.
            # It's sufficient that the metadata in the original dataset is perserved in the
            # loaded dataset.
            for k, v in md.items():  # pyright: ignore [reportAttributeAccessIssue]
                assert v == c_md[k]


@pytest.mark.skipif(maite_available, reason="MAITE is available")
@mock.patch("nrtk.interop.maite.interop.object_detection.dataset.kwcoco_available", True)
def test_missing_deps_maite() -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    assert not COCOJATICObjectDetectionDataset.is_usable()
    with pytest.raises(MaiteImportError):
        COCOJATICObjectDetectionDataset(None, list(), False)


@pytest.mark.skipif(kwcoco_available, reason="KWCOCO is available")
def test_missing_deps_kwcoco() -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    assert not COCOJATICObjectDetectionDataset.is_usable()
    with pytest.raises(KWCocoImportError):
        COCOJATICObjectDetectionDataset(None, list(), False)
