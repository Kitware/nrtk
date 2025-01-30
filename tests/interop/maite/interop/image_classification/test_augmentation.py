import copy
from collections.abc import Sequence
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from maite.protocols.image_classification import Augmentation, TargetBatchType

from nrtk.impls.perturb_image.generic.nop_perturber import NOPPerturber
from nrtk.interfaces.image_metric import ImageMetric
from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interop.maite.interop.image_classification.augmentation import (
    JATICClassificationAugmentation,
    JATICClassificationAugmentationWithMetric,
)
from tests.interop.maite.utils.test_utils import ResizePerturber

random = np.random.default_rng()


class TestJATICClassificationAugmentation:
    @pytest.mark.parametrize(
        ("perturber", "targets_in", "expected_targets_out"),
        [
            (NOPPerturber(), np.asarray([0]), np.asarray([0])),
            (ResizePerturber(w=64, h=512), np.asarray([1]), np.asarray([1])),
        ],
        ids=["no-op perturber", "resize"],
    )
    def test_augmentation_adapter(
        self,
        perturber: PerturbImage,
        targets_in: TargetBatchType,
        expected_targets_out: TargetBatchType,
    ) -> None:
        """Test that the adapter provides the same image perturbation result as the core perturber.

        Also tests that labels and metadata are appropriately updated.
        """
        augmentation = JATICClassificationAugmentation(augment=perturber)
        img_in = random.integers(0, 255, (3, 256, 256), dtype=np.uint8)  # MAITE is channels-first
        md_in: list[dict[str, Any]] = [{"some_metadata": 1}]

        # Get copies to check for modification
        img_copy = np.copy(img_in)
        targets_copy = copy.deepcopy(targets_in)
        md_copy = copy.deepcopy(md_in)

        # Get expected image and metadata from "normal" perturber
        input_image, _ = perturber(np.transpose(img_in, (1, 2, 0)))
        expected_img_out = np.transpose(input_image, (2, 0, 1))
        expected_md_out = dict(md_in[0])
        expected_md_out["nrtk::perturber"] = perturber.get_config()

        # Apply augmentation via adapter
        imgs_out, targets_out, md_out = augmentation(([img_in], targets_in, md_in))

        # Check that expectations hold
        assert np.array_equal(imgs_out[0], expected_img_out)
        assert np.array_equal(targets_out, expected_targets_out)

        for expected_tgt, tgt_out in zip(expected_targets_out, targets_out):
            assert np.array_equal(expected_tgt, tgt_out)
        assert md_out[0] == expected_md_out

        # Check that input data was not modified
        assert np.array_equal(img_in, img_copy)
        assert np.array_equal(targets_copy, targets_in)
        for tgt_copy, tgt_in in zip(targets_copy, targets_in):
            assert np.array_equal(tgt_copy, tgt_in)
        assert md_in == md_copy


class TestJATICClassificationAugmentationWithMetric:
    img_in = random.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    md_in = [{"some_metadata": 1}]
    md_aug_nop_pertuber = [
        {
            "nrtk::perturber": {"box_alignment_mode": "extent"},
            "some_metadata": 1,
        },
    ]

    @pytest.mark.parametrize(
        ("augmentations", "targets_in", "expected_targets_out", "metric_input_img2", "metric_metadata", "expectation"),
        [
            (None, [np.asarray([0])], [np.asarray([0])], None, md_in, does_not_raise()),
            (
                [JATICClassificationAugmentation(NOPPerturber())],
                [np.asarray([0])],
                [np.asarray([0])],
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
        inputs and returns the corresponding outputs in a classification augmentation
        workflow.
        """
        perturber = NOPPerturber()
        metric_patch = MagicMock(spec=ImageMetric, return_value=1.0)
        metric_augmentation = JATICClassificationAugmentationWithMetric(
            augmentations=augmentations,
            metric=metric_patch,
        )

        # Get copies to check for modification
        img_copy = np.copy(self.img_in)
        targets_copy = copy.deepcopy(targets_in)
        md_copy = copy.deepcopy(self.md_in)

        # Get expected image and metadata from "normal" perturber
        expected_img_out, _ = perturber(np.transpose(self.img_in, (2, 0, 1)))

        with expectation:
            # Apply augmentation via adapter
            imgs_out, targets_out, md_out = metric_augmentation(
                ([np.transpose(self.img_in, (2, 0, 1))], targets_in, self.md_in),
            )

            # Check if mocked metric was called with appropriate inputs
            kwargs = metric_patch.call_args.kwargs
            assert np.array_equal(kwargs["img_1"], self.img_in)
            assert np.array_equal(kwargs["img_2"], metric_input_img2)
            assert kwargs["additional_params"] == metric_metadata[0]

            # Check that expectations hold
            assert np.array_equal(imgs_out[0], expected_img_out)
            assert np.array_equal(targets_out, expected_targets_out)

            for expected_tgt, tgt_out in zip(expected_targets_out, targets_out):
                assert np.array_equal(expected_tgt, tgt_out)
            assert "nrtk::ImageMetric" in md_out[0]

            # Check that input data was not modified
            assert np.array_equal(self.img_in, img_copy)
            assert np.array_equal(targets_copy, targets_in)
            for tgt_copy, tgt_in in zip(targets_copy, targets_in):
                assert np.array_equal(tgt_copy, tgt_in)
            assert self.md_in == md_copy
