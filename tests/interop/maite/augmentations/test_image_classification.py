import copy
from collections.abc import Sequence

import numpy as np
import pytest

from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interop.maite.augmentations.object_detection import JATICDetectionAugmentation
from nrtk.interop.maite.datasets.object_detection import JATICDetectionTarget
from nrtk.utils._exceptions import MaiteImportError
from nrtk.utils._import_guard import import_guard
from nrtk.utils._nop_perturber import _NOPPerturber
from tests.interop.maite.perturber_fixtures import ResizePerturber

maite_available: bool = import_guard(module_name="maite", exception=MaiteImportError)
from maite.protocols.object_detection import DatumMetadataType, TargetType  # noqa: E402

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
        targets_in: Sequence[TargetType],  # pyright: ignore [reportInvalidTypeForm]
    ) -> None:
        """Test that the adapter appends, not overrides nrtk configs when multiple perturbations are applied."""
        img_in = random.integers(0, 255, (3, 256, 256), dtype=np.uint8)  # MAITE is channels-first
        md_in: list[DatumMetadataType] = [{"id": 1}]  # pyright: ignore [reportInvalidTypeForm]

        imgs_out = [img_in]
        targets_out = targets_in
        md_out = md_in
        for p_idx, perturber in enumerate(perturbers):
            augmentation = JATICDetectionAugmentation(augment=perturber, augment_id=f"test_augment_{p_idx}")
            imgs_out, targets_out, md_out = augmentation((imgs_out, targets_out, md_out))

        assert "nrtk_perturber_config" in md_out[0]
        all_perturber_configs = [perturber.get_config() for perturber in perturbers]
        assert md_out[0]["nrtk_perturber_config"] == all_perturber_configs
