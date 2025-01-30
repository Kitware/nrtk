import copy
from collections.abc import Sequence
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from maite.protocols.object_detection import Augmentation, TargetBatchType

from nrtk.impls.perturb_image.generic.nop_perturber import NOPPerturber
from nrtk.interfaces.image_metric import ImageMetric
from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interop.maite.interop.object_detection.augmentation import (
    JATICDetectionAugmentation,
    JATICDetectionAugmentationWithMetric,
)
from nrtk.interop.maite.interop.object_detection.dataset import JATICDetectionTarget
from tests.interop.maite.utils.test_utils import ResizePerturber

random = np.random.default_rng()


class TestJATICDetectionAugmentation:
    @pytest.mark.parametrize(
        ("perturber", "targets_in", "expected_targets_out"),
        [
            (
                NOPPerturber(),
                [
                    JATICDetectionTarget(
                        boxes=np.asarray([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]),
                        labels=np.asarray([0, 2]),
                        scores=np.asarray([0.8, 0.86]),
                    ),
                ],
                [
                    JATICDetectionTarget(
                        boxes=np.asarray([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]),
                        labels=np.asarray([0, 2]),
                        scores=np.asarray([0.8, 0.86]),
                    ),
                ],
            ),
            (
                ResizePerturber(w=64, h=512),
                [
                    JATICDetectionTarget(
                        boxes=np.asarray([[4.0, 8.0, 16.0, 32.0], [2.0, 4.0, 6.0, 8.0]]),
                        labels=np.asarray([1, 5]),
                        scores=np.asarray([0.8, 0.86]),
                    ),
                ],
                [
                    JATICDetectionTarget(
                        boxes=np.asarray([[1.0, 16.0, 4.0, 64.0], [0.5, 8.0, 1.5, 16.0]]),
                        labels=np.asarray([1, 5]),
                        scores=np.asarray([0.8, 0.86]),
                    ),
                ],
            ),
        ],
        ids=["no-op perturber", "resize"],
    )
    def test_augmentation_adapter(
        self,
        perturber: PerturbImage,
        targets_in: TargetBatchType,
        expected_targets_out: TargetBatchType,
    ) -> None:
        """Test that the augmentation adapter functions appropriately.

        Tests that the adapter provides the same image perturbation result
        as the core perturber and that bboxes and metadata are appropriately
        updated.
        """
        augmentation = JATICDetectionAugmentation(augment=perturber)
        img_in = random.integers(0, 255, (3, 256, 256), dtype=np.uint8)
        md_in: list[dict[str, Any]] = [{"some_metadata": 1}]

        # Get copies to check for modification
        img_copy = np.copy(img_in)
        targets_copy = copy.deepcopy(targets_in)
        md_copy = copy.deepcopy(md_in)

        # Get expected image and metadata from "normal" perturber
        expected_img_out, _ = perturber(np.transpose(img_in, (1, 2, 0)))
        # switch from channel last to channel first
        expected_img_out = np.transpose(expected_img_out, (2, 0, 1))
        expected_md_out = dict(md_in[0])
        expected_md_out["nrtk::perturber"] = perturber.get_config()

        # Apply augmentation via adapter
        imgs_out, targets_out, md_out = augmentation(([img_in], targets_in, md_in))

        # Check that expectations hold
        assert np.array_equal(imgs_out[0], expected_img_out)
        assert len(targets_out) == len(expected_targets_out)
        for expected_tgt, tgt_out in zip(expected_targets_out, targets_out):
            assert np.array_equal(expected_tgt.boxes, tgt_out.boxes)
            assert np.array_equal(expected_tgt.labels, tgt_out.labels)
            assert np.array_equal(expected_tgt.scores, tgt_out.scores)
        assert md_out[0] == expected_md_out

        # Check that input data was not modified
        assert np.array_equal(img_in, img_copy)
        assert len(targets_copy) == len(targets_in)
        for tgt_copy, tgt_in in zip(targets_copy, targets_in):
            assert np.array_equal(tgt_copy.boxes, tgt_in.boxes)
            assert np.array_equal(tgt_copy.labels, tgt_in.labels)
            assert np.array_equal(tgt_copy.scores, tgt_in.scores)
        assert md_in == md_copy


class TestJATICDetectionAugmentationWithMetric:
    img_in = random.integers(0, 255, (3, 256, 256), dtype=np.uint8)
    md_in = [{"some_metadata": 1}]
    md_aug_nop_pertuber = [{"nrtk::perturber": {"box_alignment_mode": "extent"}, "some_metadata": 1}]

    @pytest.mark.parametrize(
        ("augmentations", "targets_in", "expected_targets_out", "metric_input_img2", "metric_metadata", "expectation"),
        [
            (
                None,
                [
                    JATICDetectionTarget(
                        boxes=np.asarray([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]),
                        labels=np.asarray([0, 2]),
                        scores=np.asarray([0.8, 0.86]),
                    ),
                ],
                [
                    JATICDetectionTarget(
                        boxes=np.asarray([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]),
                        labels=np.asarray([0, 2]),
                        scores=np.asarray([0.8, 0.86]),
                    ),
                ],
                None,
                md_in,
                does_not_raise(),
            ),
            (
                [JATICDetectionAugmentation(NOPPerturber())],
                [
                    JATICDetectionTarget(
                        boxes=np.asarray([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]),
                        labels=np.asarray([0, 2]),
                        scores=np.asarray([0.8, 0.86]),
                    ),
                ],
                [
                    JATICDetectionTarget(
                        boxes=np.asarray([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]),
                        labels=np.asarray([0, 2]),
                        scores=np.asarray([0.8, 0.86]),
                    ),
                ],
                img_in,
                md_aug_nop_pertuber,
                does_not_raise(),
            ),
        ],
        ids=["None", "no-op perturber"],
    )
    def test_metric_augmentation_adapter(
        self,
        augmentations: Sequence[Augmentation],
        targets_in: TargetBatchType,
        expected_targets_out: TargetBatchType,
        metric_input_img2: np.ndarray,
        metric_metadata: list[dict[str, Any]],
        expectation: AbstractContextManager,
    ) -> None:
        """Test that the augmentation adapter works with the Image Metric workflow.

        Tests that the adapter calls the metric computation with the the appropriate
        inputs and returns the corresponding outputs in a detection augmentation
        workflow.
        """
        perturber = NOPPerturber()
        metric_patch = MagicMock(spec=ImageMetric, return_value=1.0)
        metric_augmentation = JATICDetectionAugmentationWithMetric(augmentations=augmentations, metric=metric_patch)

        # Get copies to check for modification
        img_copy = np.copy(self.img_in)
        targets_copy = copy.deepcopy(targets_in)
        md_copy = copy.deepcopy(self.md_in)

        # Get expected image and metadata from "normal" perturber
        expected_img_out, _ = perturber(self.img_in)

        with expectation:
            # Apply augmentation via adapter
            imgs_out, targets_out, md_out = metric_augmentation(([self.img_in], targets_in, self.md_in))

            # Check if mocked metric was called with appropriate inputs
            kwargs = metric_patch.call_args.kwargs
            assert np.array_equal(kwargs["img_1"], self.img_in)
            assert np.array_equal(kwargs["img_2"], metric_input_img2)
            assert kwargs["additional_params"] == metric_metadata[0]

            # Check that expectations hold
            assert np.array_equal(imgs_out[0], expected_img_out)
            assert len(targets_out) == len(expected_targets_out)
            for expected_tgt, tgt_out in zip(expected_targets_out, targets_out):
                assert np.array_equal(expected_tgt.boxes, tgt_out.boxes)
                assert np.array_equal(expected_tgt.labels, tgt_out.labels)
                assert np.array_equal(expected_tgt.scores, tgt_out.scores)
            assert "nrtk::ImageMetric" in md_out[0]

            # Check that input data was not modified
            assert np.array_equal(self.img_in, img_copy)
            assert len(targets_copy) == len(targets_in)
            for tgt_copy, tgt_in in zip(targets_copy, targets_in):
                assert np.array_equal(tgt_copy.boxes, tgt_in.boxes)
                assert np.array_equal(tgt_copy.labels, tgt_in.labels)
                assert np.array_equal(tgt_copy.scores, tgt_in.scores)
            assert self.md_in == md_copy
