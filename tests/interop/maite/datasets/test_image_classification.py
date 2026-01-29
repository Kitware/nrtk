from collections.abc import Sequence

import numpy as np
import pytest

from nrtk.interop import MAITEImageClassificationAugmentation
from nrtk.interop._maite.datasets.image_classification import (
    MAITEImageClassificationDataset,
)
from nrtk.utils._exceptions import MaiteImportError
from nrtk.utils._import_guard import import_guard
from tests.interop.maite.perturber_fixtures import ResizePerturber

maite_available: bool = import_guard(module_name="maite", exception=MaiteImportError)
from maite.protocols.image_classification import TargetType  # noqa: E402

random = np.random.default_rng()


class TestMAITEImageClassificationDataset:
    @pytest.mark.skipif(not maite_available, reason=str(MaiteImportError()))
    @pytest.mark.parametrize(
        ("dataset", "expected_lbls_out"),
        [
            (
                MAITEImageClassificationDataset(
                    imgs=[
                        random.integers(0, 255, (3, 256, 256), dtype=np.uint8),
                        random.integers(0, 255, (3, 128, 128), dtype=np.uint8),
                    ],
                    labels=[np.asarray([0]), np.asarray([1])],
                    datum_metadata=[{"id": 0}, {"id": 1}],
                    dataset_id="dummy_dataset",
                    index2label={0: "cat0", 1: "cat1"},
                ),
                [np.asarray([0]), np.asarray([1])],
            ),
            (
                MAITEImageClassificationDataset(
                    imgs=[
                        random.integers(0, 255, (3, 256, 256), dtype=np.uint8),
                        random.integers(0, 255, (3, 128, 128), dtype=np.uint8),
                    ],
                    labels=[np.asarray([0]), np.asarray([1])],
                    datum_metadata=[{"id": 0}, {"id": 1}],
                    dataset_id="dummy_dataset",
                ),
                [np.asarray([0]), np.asarray([1])],
            ),
        ],
    )
    def test_dataset_adapter(
        self,
        dataset: MAITEImageClassificationDataset,
        expected_lbls_out: Sequence[TargetType],  # pyright: ignore [reportInvalidTypeForm]
    ) -> None:
        """Test that the dataset adapter functions appropriately.

        Tests that the dataset adapter takes in an input of varying size images with corresponding labels
        and metadata and can be ingested by the augmentation adapter object.
        """
        perturber = ResizePerturber(w=64, h=512)
        augmentation = MAITEImageClassificationAugmentation(augment=perturber, augment_id="test_augment")
        for idx in range(len(dataset)):
            img_in = dataset[idx][0]
            lbl_in = dataset[idx][1]
            md_in = dataset[idx][2]

            # Get expected image and metadata from "normal" perturber
            input_image, _ = perturber(image=np.transpose(np.asarray(img_in), (1, 2, 0)))
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
