import copy
from collections.abc import Sequence

import numpy as np
import pytest
from maite.protocols.image_classification import DatumMetadataType

from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interop import MAITEImageClassificationAugmentation
from tests.fakes import FakePerturber
from tests.interop.maite.perturber_fixtures import ResizePerturber

random = np.random.default_rng()


@pytest.mark.maite
class TestMAITEImageClassificationAugmentation:
    @pytest.mark.parametrize(
        "perturber",
        [
            FakePerturber(),
            ResizePerturber(w=64, h=512),
        ],
        ids=["no-op perturber", "resize"],
    )
    def test_augmentation_adapter(
        self,
        perturber: PerturbImage,
    ) -> None:
        """Test that the augmentation adapter functions appropriately.

        Tests that the adapter generates the same image perturbation result
        as the core perturber and that bboxes and metadata are appropriately
        updated.
        """
        augmentation = MAITEImageClassificationAugmentation(augment=perturber, augment_id="test_augment")
        img_in = random.integers(0, 255, (3, 256, 256), dtype=np.uint8)
        target_class_in = [0]
        md_in: list[DatumMetadataType] = [{"id": 1}]  # pyright: ignore [reportInvalidTypeForm]

        # Get copies to check for modification
        img_copy = np.copy(img_in)
        target_class_copy = copy.deepcopy(target_class_in)
        md_copy = copy.deepcopy(md_in)

        # Get expected image and metadata from "normal" perturber
        expected_img_out, _ = perturber(image=np.transpose(img_in, (1, 2, 0)))
        # switch from channel last to channel first
        expected_img_out = np.transpose(expected_img_out, (2, 0, 1))
        expected_target_class = [0]
        expected_md_out = dict(md_in[0])
        expected_md_out["nrtk_perturber_config"] = [perturber.get_config()]

        # Apply augmentation via adapter
        imgs_out, target_class_out, md_out = augmentation(([img_in], target_class_copy, md_in))

        # Check that expectations hold
        assert np.array_equal(imgs_out[0], expected_img_out)
        assert len(target_class_out) == len(expected_target_class)
        assert target_class_out == expected_target_class
        assert md_out[0] == expected_md_out

        # Check that input data was not modified
        assert np.array_equal(img_in, img_copy)
        assert len(target_class_copy) == len(target_class_in)
        assert target_class_copy == target_class_in
        assert md_in == md_copy

    @pytest.mark.parametrize(
        "perturbers",
        [
            [FakePerturber(), ResizePerturber(w=64, h=512)],
        ],
    )
    def test_multiple_augmentations(
        self,
        perturbers: Sequence[PerturbImage],
    ) -> None:
        """Test that the adapter appends, not overrides nrtk configs when multiple perturbations are applied."""
        img_in = random.integers(0, 255, (3, 256, 256), dtype=np.uint8)  # MAITE is channels-first
        targets_in = [0]
        md_in: list[DatumMetadataType] = [{"id": 1}]  # pyright: ignore [reportInvalidTypeForm]

        imgs_out = [img_in]
        targets_out = targets_in
        md_out = md_in
        for p_idx, perturber in enumerate(perturbers):
            augmentation = MAITEImageClassificationAugmentation(augment=perturber, augment_id=f"test_augment_{p_idx}")
            imgs_out, targets_out, md_out = augmentation((imgs_out, targets_out, md_out))

        assert "nrtk_perturber_config" in md_out[0]
        all_perturber_configs = [perturber.get_config() for perturber in perturbers]
        assert md_out[0].get("nrtk_perturber_config") == all_perturber_configs
