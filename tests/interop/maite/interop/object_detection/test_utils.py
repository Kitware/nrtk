import json
import unittest.mock as mock
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import py  # type: ignore
import pytest
from maite.protocols.object_detection import Dataset

from nrtk.interop.maite.interop.object_detection.dataset import (
    JATICDetectionTarget,
    JATICObjectDetectionDataset,
)
from nrtk.interop.maite.interop.object_detection.utils import is_usable

if is_usable:
    # Multiple type ignores added for pyright's handling of guarded imports
    import kwcoco  # type: ignore

from nrtk.interop.maite.interop.object_detection.dataset import (
    COCOJATICObjectDetectionDataset,
)
from nrtk.interop.maite.interop.object_detection.utils import dataset_to_coco
from nrtk.utils._exceptions import KWCocoImportError

random = np.random.default_rng()


@pytest.mark.skipif(not is_usable, reason=str(KWCocoImportError()))
@pytest.mark.parametrize(
    ("dataset", "img_filenames", "categories", "expectation"),
    [
        (
            JATICObjectDetectionDataset(
                imgs=[random.integers(0, 255, size=(3, 10, 10), dtype=np.uint8)],
                dets=[
                    JATICDetectionTarget(
                        boxes=random.integers(0, 4, size=(2, 4)),
                        labels=random.integers(0, 2, size=(2,)),
                        scores=random.random(2),
                    ),
                ],
                datum_metadata=[{"id": 0}],
                dataset_id="dummy_dataset",
            ),
            ["images/img1.png"],
            [
                {"id": 0, "name": "cat0", "supercategory": None},
                {"id": 1, "name": "cat1", "supercategory": None},
            ],
            does_not_raise(),
        ),
        (
            JATICObjectDetectionDataset(
                imgs=[random.integers(0, 255, size=(3, 3, 3), dtype=np.uint8)] * 2,
                dets=[
                    JATICDetectionTarget(
                        boxes=random.integers(0, 4, size=(2, 4)),
                        labels=random.integers(0, 2, size=(2,)),
                        scores=random.random(2),
                    ),
                ]
                * 2,
                datum_metadata=[{"id": idx} for idx in range(2)],
                dataset_id="dummy_dataset",
            ),
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
    dataset: Dataset,
    img_filenames: list[Path],
    categories: list[dict[str, Any]],
    expectation: AbstractContextManager,
    tmpdir: py.path.local,
) -> None:
    """Test that a MAITE dataset is appropriately saved to file in COCO format."""
    tmp_path = Path(tmpdir)

    with expectation:
        dataset_to_coco(
            dataset=dataset,
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

            assert np.array_equal(image, c_image)
            assert np.array_equal(dets.boxes, c_dets.boxes)
            assert np.array_equal(dets.labels, c_dets.labels)
            # Not checking scores as they are not perserved

            # Not checking for total equality because the COCO dataset class adds metadata.
            # It's sufficient that the metadata in the original dataset is perserved in the
            # loaded dataset.
            for k, v in md.items():
                assert v == c_md[k]


@mock.patch.object(COCOJATICObjectDetectionDataset, "is_usable")
def test_missing_deps(mock_is_usable: MagicMock) -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    mock_is_usable.return_value = False
    assert not COCOJATICObjectDetectionDataset.is_usable()
    with pytest.raises(KWCocoImportError):
        COCOJATICObjectDetectionDataset(None, list(), False)
