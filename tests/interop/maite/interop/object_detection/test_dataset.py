from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest

from nrtk.interop.maite.interop.object_detection.augmentation import JATICDetectionAugmentation
from nrtk.interop.maite.interop.object_detection.dataset import JATICObjectDetectionDataset
from nrtk.interop.maite.interop.object_detection.utils import maite_available
from nrtk.utils._exceptions import MaiteImportError
from tests.interop.maite.utils.test_utils import ResizePerturber

TargetType: type = object
DatumMetadataType: type = object
if maite_available:
    from maite.protocols.object_detection import DatumMetadataType, TargetType

random = np.random.default_rng()


@dataclass
class JATICDetectionTarget:
    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


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
            expected_img_out, _ = perturber(np.transpose(np.asarray(img_in), (1, 2, 0)))
            # Channel last to channel first
            expected_img_out = np.transpose(expected_img_out, (2, 0, 1))
            expected_md_out = dict(md_in)  # pyright: ignore [reportArgumentType, reportCallIssue]
            expected_md_out["nrtk_perturber_config"] = [perturber.get_config()]  # pyright: ignore [reportArgumentType]

            # Apply augmentation via adapter
            img_out, det_out, md_out = augmentation(([img_in], [det_in], [md_in]))
            expected_det_out = expected_dets_out[idx]

            # Check that expectations hold
            assert np.array_equal(img_out[0], expected_img_out)
            for det, target in zip(det_out, expected_det_out):
                assert np.array_equal(det.boxes, target.boxes)
                assert np.array_equal(det.labels, target.labels)
                assert np.array_equal(det.scores, target.scores)
            assert md_out[0] == expected_md_out
