import copy
from collections.abc import Sequence
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from maite.protocols.object_detection import DatumMetadataType, TargetType

from nrtk.interfaces.image_metric import ImageMetric
from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interop.maite.augmentations.object_detection import (
    JATICDetectionAugmentation,
    JATICDetectionAugmentationWithMetric,
)
from nrtk.interop.maite.datasets.object_detection import JATICDetectionTarget
from nrtk.interop.maite.utils.detection import maite_available
from nrtk.utils._exceptions import MaiteImportError
from nrtk.utils._nop_perturber import _NOPPerturber
from tests.interop.maite.utils.test_utils import ResizePerturber

random = np.random.default_rng()


@pytest.mark.skipif(not maite_available, reason=str(MaiteImportError()))
class TestJATICDetectionAugmentation:
    @pytest.mark.parametrize(
        ("perturber", "targets_in", "expected_targets_out"),
        [
            (
                _NOPPerturber(),
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
        targets_in: Sequence[TargetType],  # pyright: ignore [reportInvalidTypeForm]
        expected_targets_out: Sequence[TargetType],  # pyright: ignore [reportInvalidTypeForm]
    ) -> None:
        """Test that the augmentation adapter functions appropriately.

        Tests that the adapter generates the same image perturbation result
        as the core perturber and that bboxes and metadata are appropriately
        updated.
        """
        augmentation = JATICDetectionAugmentation(augment=perturber, augment_id="test_augment")
        img_in = random.integers(0, 255, (3, 256, 256), dtype=np.uint8)
        md_in: list[DatumMetadataType] = [{"id": 1}]  # pyright: ignore [reportInvalidTypeForm]

        # Get copies to check for modification
        img_copy = np.copy(img_in)
        targets_copy = copy.deepcopy(targets_in)
        md_copy = copy.deepcopy(md_in)

        # Get expected image and metadata from "normal" perturber
        expected_img_out, _ = perturber(image=np.transpose(img_in, (1, 2, 0)))
        # switch from channel last to channel first
        expected_img_out = np.transpose(expected_img_out, (2, 0, 1))
        expected_md_out = dict(md_in[0])
        expected_md_out["nrtk_perturber_config"] = [perturber.get_config()]

        # Apply augmentation via adapter
        imgs_out, targets_out, md_out = augmentation(([img_in], targets_in, md_in))

        # Check that expectations hold
        assert np.array_equal(imgs_out[0], expected_img_out)
        assert len(targets_out) == len(expected_targets_out)
        for expected_tgt, tgt_out in zip(expected_targets_out, targets_out, strict=False):
            assert np.array_equal(expected_tgt.boxes, tgt_out.boxes)
            assert np.array_equal(expected_tgt.labels, tgt_out.labels)
            assert np.array_equal(expected_tgt.scores, tgt_out.scores)
        assert md_out[0] == expected_md_out

        # Check that input data was not modified
        assert np.array_equal(img_in, img_copy)
        assert len(targets_copy) == len(targets_in)
        for tgt_copy, tgt_in in zip(targets_copy, targets_in, strict=False):
            assert np.array_equal(tgt_copy.boxes, tgt_in.boxes)
            assert np.array_equal(tgt_copy.labels, tgt_in.labels)
            assert np.array_equal(tgt_copy.scores, tgt_in.scores)
        assert md_in == md_copy

    @pytest.mark.parametrize(
        ("perturbers", "targets_in"),
        [
            (
                [_NOPPerturber(), ResizePerturber(w=64, h=512)],
                [
                    JATICDetectionTarget(
                        boxes=np.asarray([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]),
                        labels=np.asarray([0, 2]),
                        scores=np.asarray([0.8, 0.86]),
                    ),
                ],
            ),
        ],
    )
    def test_multiple_augmentations(
        self,
        perturbers: Sequence[PerturbImage],
        targets_in: Sequence[TargetType],
    ) -> None:
        """Test that the adapter appends, not overrides nrtk configs when multiple perturbations are applied."""
        img_in = random.integers(0, 255, (3, 256, 256), dtype=np.uint8)  # MAITE is channels-first
        md_in: list[DatumMetadataType] = [{"id": 1}]

        imgs_out = [img_in]
        targets_out = targets_in
        md_out = md_in
        for p_idx, perturber in enumerate(perturbers):
            augmentation = JATICDetectionAugmentation(augment=perturber, augment_id=f"test_augment_{p_idx}")
            imgs_out, targets_out, md_out = augmentation((imgs_out, targets_out, md_out))

        assert "nrtk_perturber_config" in md_out[0]
        all_perturber_configs = [perturber.get_config() for perturber in perturbers]
        assert md_out[0]["nrtk_perturber_config"] == all_perturber_configs


@pytest.mark.skipif(not maite_available, reason=str(MaiteImportError()))
class TestJATICDetectionAugmentationWithMetric:
    img_in = random.integers(0, 255, (3, 256, 256), dtype=np.uint8)
    md_in: list["DatumMetadataType"] = [{"id": 1}]
    md_aug_nop_pertuber = [{"nrtk_perturber_config": [{"theta": 0}], "id": 1}]

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
                [(_NOPPerturber(), "no op augment")],
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
        augmentations: Sequence[tuple[PerturbImage, str]],  # pyright: ignore [reportPossiblyUnboundVariable]
        targets_in: Sequence[TargetType],  # pyright: ignore [reportInvalidTypeForm]
        expected_targets_out: Sequence[TargetType],  # pyright: ignore [reportInvalidTypeForm]
        metric_input_img2: np.ndarray,
        metric_metadata: list[dict[str, Any]],
        expectation: AbstractContextManager,
    ) -> None:
        """Test that the augmentation adapter works with the Image Metric workflow.

        Tests that the adapter calls the metric computation with the the appropriate
        inputs and returns the corresponding outputs in a detection augmentation
        workflow.
        """
        perturber = _NOPPerturber()
        metric_patch = MagicMock(spec=ImageMetric, return_value=1.0)
        metric_augmentation = JATICDetectionAugmentationWithMetric(
            augmentations=[
                JATICDetectionAugmentation(augment=augment, augment_id=idx) for augment, idx in augmentations
            ]
            if augmentations is not None
            else None,
            metric=metric_patch,
            augment_id="test_augment_with_metric",
        )

        # Get copies to check for modification
        img_copy = np.copy(self.img_in)
        targets_copy = copy.deepcopy(targets_in)
        md_copy = copy.deepcopy(self.md_in)

        # Get expected image and metadata from "normal" perturber
        expected_img_out, _ = perturber(image=self.img_in)

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
            for expected_tgt, tgt_out in zip(expected_targets_out, targets_out, strict=False):
                assert np.array_equal(expected_tgt.boxes, tgt_out.boxes)
                assert np.array_equal(expected_tgt.labels, tgt_out.labels)
                assert np.array_equal(expected_tgt.scores, tgt_out.scores)
            assert "nrtk_metric" in md_out[0]

            # Check that input data was not modified
            assert np.array_equal(self.img_in, img_copy)
            assert len(targets_copy) == len(targets_in)
            for tgt_copy, tgt_in in zip(targets_copy, targets_in, strict=False):
                assert np.array_equal(tgt_copy.boxes, tgt_in.boxes)
                assert np.array_equal(tgt_copy.labels, tgt_in.labels)
                assert np.array_equal(tgt_copy.scores, tgt_in.scores)
            assert self.md_in == md_copy

    @pytest.mark.parametrize(
        ("augmentations", "targets_in"),
        [
            (
                [(_NOPPerturber(), "no op augment")],
                [
                    JATICDetectionTarget(
                        boxes=np.asarray([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]),
                        labels=np.asarray([0, 2]),
                        scores=np.asarray([0.8, 0.86]),
                    ),
                ],
            ),
        ],
    )
    def test_multiple_metrics(
        self,
        augmentations: Sequence[tuple[PerturbImage, str]],
        targets_in: Sequence[TargetType],
    ) -> None:
        """Test that multiple metrics can be added to metadata."""
        imgs_out = [np.transpose(self.img_in, (2, 0, 1))]
        targets_out = targets_in
        md_out = self.md_in

        num_augments = 2
        metric_patches = [MagicMock(spec=ImageMetric, return_value=idx) for idx in range(num_augments)]
        for idx in range(num_augments):
            metric_augmentation = JATICDetectionAugmentationWithMetric(
                augmentations=[
                    JATICDetectionAugmentation(augment=augment, augment_id=idx) for augment, idx in augmentations
                ],
                metric=metric_patches[idx],
                augment_id=f"test_augment_with_metric{1}",
            )

            # Apply augmentation via adapter
            imgs_out, targets_out, md_out = metric_augmentation(
                (imgs_out, targets_out, md_out),
            )

        assert "nrtk_metric" in md_out[0]
        assert md_out[0]["nrtk_metric"] == [("ImageMetric", idx) for idx in range(num_augments)]
