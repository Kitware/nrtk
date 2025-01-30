from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest
from maite.protocols.object_detection import TargetType

from nrtk.interop.maite.interop.object_detection.augmentation import JATICDetectionAugmentation
from nrtk.interop.maite.interop.object_detection.dataset import JATICObjectDetectionDataset
from tests.interop.maite.utils.test_utils import ResizePerturber

random = np.random.default_rng()


@dataclass
class JATICDetectionTarget:
    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


class TestJATICImageClassificationDataset:
    @pytest.mark.parametrize(
        ("dataset", "expected_dets_out"),
        [
            (
                JATICObjectDetectionDataset(
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
                    [{"some_metadata": 0}, {"some_metadata": 1}],
                ),
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
        dataset: JATICObjectDetectionDataset,
        expected_dets_out: Sequence[Sequence[TargetType]],
    ) -> None:
        """Test that the dataset adapter performs appropriately.

        Test that the dataset adapter takes in an input of varying size
        images with corresponding detections and metadata and can be ingested
        by the augmentation adapter object.
        """
        perturber = ResizePerturber(w=64, h=512)
        augmentation = JATICDetectionAugmentation(augment=perturber)
        for idx in range(len(dataset)):
            img_in = dataset[idx][0]
            det_in = dataset[idx][1]
            md_in = dataset[idx][2]

            # Get expected image and metadata from "normal" perturber
            expected_img_out, _ = perturber(np.transpose(np.asarray(img_in), (1, 2, 0)))
            # Channel last to channel first
            expected_img_out = np.transpose(expected_img_out, (2, 0, 1))
            expected_md_out = dict(md_in)
            expected_md_out["nrtk::perturber"] = perturber.get_config()

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
