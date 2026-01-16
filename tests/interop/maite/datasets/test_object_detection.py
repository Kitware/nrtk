import json
import unittest.mock as mock
from collections.abc import Sequence
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Any

import numpy as np
import py
import pytest

from nrtk.interop.maite.augmentations.object_detection import JATICDetectionAugmentation
from nrtk.interop.maite.datasets.object_detection import (
    COCOJATICObjectDetectionDataset,
    JATICDetectionTarget,
    JATICObjectDetectionDataset,
    dataset_to_coco,
)
from nrtk.utils._exceptions import KWCocoImportError, MaiteImportError
from nrtk.utils._import_guard import import_guard
from tests.interop.maite.perturber_fixtures import ResizePerturber

kwcoco_available: bool = import_guard(module_name="kwcoco", exception=KWCocoImportError)
maite_available: bool = import_guard(module_name="maite", exception=MaiteImportError)
import kwcoco  # noqa: E402
from maite.protocols.object_detection import DatumMetadataType, TargetType  # noqa: E402

random = np.random.default_rng()


@pytest.mark.skipif(not maite_available, reason=str(MaiteImportError()))
class TestJATICObjectDetectionDataset:
    @pytest.mark.parametrize(
        ("imgs", "input_dets", "datum_metadata", "dataset_id", "index2label", "expected_dets_out"),
        [
            (
                [
                    random.integers(0, 255, (3, 256, 256), dtype=np.uint8),
                    random.integers(0, 255, (3, 128, 128), dtype=np.uint8),
                ],
                [
                    JATICDetectionTarget(
                        boxes=np.asarray([[0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 100.0, 100.0]]),
                        labels=np.asarray([1, 1]),
                        scores=np.asarray([1, 1]),
                    ),
                    JATICDetectionTarget(
                        boxes=np.asarray([[0.0, 0, 50.0, 50.0]]),
                        labels=np.asarray([0]),
                        scores=np.asarray([1]),
                    ),
                ],
                [{"id": 0}, {"id": 1}],
                "dummy_dataset",
                {0: "cat1", 1: "cat2"},
                [
                    [
                        JATICDetectionTarget(
                            boxes=np.asarray([[0.0, 0.0, 25.0, 200.0], [0.0, 0.0, 25.0, 200.0]]),
                            labels=np.asarray([1, 1]),
                            scores=np.asarray([1, 1]),
                        ),
                    ],
                    [
                        JATICDetectionTarget(
                            boxes=np.asarray([[0.0, 0, 25.0, 200.0]]),
                            labels=np.asarray([0]),
                            scores=np.asarray([1]),
                        ),
                    ],
                ],
            ),
        ],
    )
    def test_dataset_adapter(
        self,
        imgs: Sequence[np.ndarray],
        input_dets: Sequence[TargetType],  # pyright: ignore [reportInvalidTypeForm]
        datum_metadata: Sequence[DatumMetadataType],  # pyright: ignore [reportInvalidTypeForm]
        dataset_id: str,
        index2label: dict[int, str],
        expected_dets_out: Sequence[Sequence[TargetType]],  # pyright: ignore [reportInvalidTypeForm]
    ) -> None:
        """Test that the dataset adapter performs appropriately.

        Test that the dataset adapter takes in an input of varying size
        images with corresponding detections and metadata and can be ingested
        by the augmentation adapter object.
        """
        dataset = JATICObjectDetectionDataset(
            imgs=imgs,
            dets=input_dets,
            datum_metadata=datum_metadata,
            dataset_id=dataset_id,
            index2label=index2label,
        )
        perturber = ResizePerturber(w=64, h=512)
        augmentation = JATICDetectionAugmentation(augment=perturber, augment_id="test_augment")
        for idx in range(len(dataset)):
            img_in = dataset[idx][0]
            det_in = dataset[idx][1]
            md_in = dataset[idx][2]

            # Get expected image and metadata from "normal" perturber
            expected_img_out, _ = perturber(image=np.transpose(np.asarray(img_in), (1, 2, 0)))
            # Channel last to channel first
            expected_img_out = np.transpose(expected_img_out, (2, 0, 1))
            expected_md_out = dict(md_in)
            expected_md_out["nrtk_perturber_config"] = [perturber.get_config()]

            # Apply augmentation via adapter
            img_out, det_out, md_out = augmentation(([img_in], [det_in], [md_in]))
            expected_det_out = expected_dets_out[idx]

            # Check that expectations hold
            assert np.array_equal(img_out[0], expected_img_out)
            for det, target in zip(det_out, expected_det_out, strict=False):
                assert np.array_equal(det.boxes, target.boxes)
                assert np.array_equal(det.labels, target.labels)
                assert np.array_equal(det.scores, target.scores)
            assert md_out[0] == expected_md_out


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
            kwcoco_dataset=kwcoco.CocoDataset(label_file),  # pyright: ignore [reportCallIssue]
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
@mock.patch("nrtk.interop.maite.datasets.object_detection.kwcoco_available", True)
def test_missing_deps_maite() -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    assert not COCOJATICObjectDetectionDataset.is_usable()
    with pytest.raises(MaiteImportError):
        COCOJATICObjectDetectionDataset(kwcoco_dataset=None, image_metadata=list(), skip_no_anns=False)


@pytest.mark.skipif(kwcoco_available, reason="KWCOCO is available")
def test_missing_deps_kwcoco() -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    assert not COCOJATICObjectDetectionDataset.is_usable()
    with pytest.raises(KWCocoImportError):
        COCOJATICObjectDetectionDataset(kwcoco_dataset=None, image_metadata=list(), skip_no_anns=False)
