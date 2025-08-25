from collections.abc import Sequence

import numpy as np
import pytest
from maite.protocols.image_classification import TargetType

from nrtk.interop.maite.interop.image_classification.augmentation import (
    JATICClassificationAugmentation,
)
from nrtk.interop.maite.interop.image_classification.dataset import (
    JATICImageClassificationDataset,
)
from nrtk.utils._exceptions import MaiteImportError
from nrtk.utils._import_guard import is_available
from tests.interop.maite.utils.test_utils import ResizePerturber

maite_available: bool = is_available("maite")

random = np.random.default_rng()


class TestJATICImageClassificationDataset:
    @pytest.mark.skipif(not maite_available, reason=str(MaiteImportError()))
    @pytest.mark.parametrize(
        ("dataset", "expected_lbls_out"),
        [
            (
                JATICImageClassificationDataset(
                    [
                        random.integers(0, 255, (3, 256, 256), dtype=np.uint8),
                        random.integers(0, 255, (3, 128, 128), dtype=np.uint8),
                    ],
                    [np.asarray([0]), np.asarray([1])],
                    [{"id": 0}, {"id": 1}],
                    "dummy_dataset",
                    {0: "cat0", 1: "cat1"},
                ),
                [np.asarray([0]), np.asarray([1])],
            ),
            (
                JATICImageClassificationDataset(
                    [
                        random.integers(0, 255, (3, 256, 256), dtype=np.uint8),
                        random.integers(0, 255, (3, 128, 128), dtype=np.uint8),
                    ],
                    [np.asarray([0]), np.asarray([1])],
                    [{"id": 0}, {"id": 1}],
                    "dummy_dataset",
                ),
                [np.asarray([0]), np.asarray([1])],
            ),
        ],
    )
    def test_dataset_adapter(
        self,
        dataset: JATICImageClassificationDataset,
        expected_lbls_out: Sequence[TargetType],  # pyright: ignore [reportInvalidTypeForm]
    ) -> None:
        """Test that the dataset adapter functions appropriately.

        Tests that the dataset adapter takes in an input of varying size images with corresponding labels
        and metadata and can be ingested by the augmentation adapter object.
        """
        perturber = ResizePerturber(w=64, h=512)
        augmentation = JATICClassificationAugmentation(augment=perturber, augment_id="test_augment")
        for idx in range(len(dataset)):
            img_in = dataset[idx][0]
            lbl_in = dataset[idx][1]
            md_in = dataset[idx][2]

            # Get expected image and metadata from "normal" perturber
            input_image, _ = perturber(np.transpose(np.asarray(img_in), (1, 2, 0)))
            expected_img_out = np.transpose(input_image, (2, 0, 1))
            expected_md_out = dict(md_in)
            expected_md_out["nrtk_perturber_config"] = [perturber.get_config()]

            # Apply augmentation via adapter
            img_out, lbl_out, md_out = augmentation(([img_in], [lbl_in], [md_in]))
            expected_lbl_out = expected_lbls_out[idx]

            # Check that expectations hold
            assert np.array_equal(img_out[0], expected_img_out)
            assert np.array_equal(lbl_out[0], expected_lbl_out)
            assert md_out[0] == expected_md_out
