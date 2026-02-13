from collections.abc import Hashable, Sequence
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any

import numpy as np
import pytest
from PIL import Image
from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from syrupy.assertion import SnapshotAssertion

from nrtk.impls.perturb_image.geometric.random import RandomRotationPerturber
from tests.impls import INPUT_TANK_IMG_FILE_PATH as INPUT_IMG_FILE_PATH
from tests.impls.perturb_image.perturber_tests_mixin import PerturberTestsMixin
from tests.impls.perturb_image.perturber_utils import perturber_assertions

rng = np.random.default_rng()


@pytest.mark.opencv
@pytest.mark.albumentations
class TestRandomRotationPerturber(PerturberTestsMixin):
    impl_class = RandomRotationPerturber

    @pytest.mark.parametrize(
        ("limit", "probability", "fill", "expectation"),
        [
            (
                90,
                -0.5,
                [255, 0, 0],
                pytest.raises(
                    ValueError,
                    match=r"Rotation probability must be between 0.0 and 1.0 inclusive.",
                ),
            ),
            (
                90,
                1.5,
                [255, 0, 0],
                pytest.raises(
                    ValueError,
                    match=r"Rotation probability must be between 0.0 and 1.0 inclusive.",
                ),
            ),
            (
                0,
                1.0,
                [355, 0, 0],
                pytest.raises(
                    ValueError,
                    match=r"Color fill values must be integers between 0 and 255 inclusive.",
                ),
            ),
            (
                0,
                1.0,
                [-155, 0, 0],
                pytest.raises(
                    ValueError,
                    match=r"Color fill values must be integers between 0 and 255 inclusive.",
                ),
            ),
            (0, 1.0, [255, 0, 0], does_not_raise()),
            (150, 0.0, [255, 0, 0], does_not_raise()),
            (120, 0.5, [255, 0, 0], does_not_raise()),
            (500, 1.0, [255, 0, 0], does_not_raise()),
            ((0, 90), 1.0, [255, 0, 0], does_not_raise()),
            ((-90, 0), 1.0, [255, 0, 0], does_not_raise()),
            ((-180, 270), 1.0, [255, 0, 0], does_not_raise()),
        ],
    )
    def test_perturber_validity(
        self,
        limit: float | tuple[float, float],
        probability: float,
        fill: Sequence[int],
        expectation: AbstractContextManager,
    ) -> None:
        """Raise appropriate errors for invalid perturber arguments."""
        with expectation:
            RandomRotationPerturber(
                limit=limit,
                probability=probability,
                fill=fill,
            )

    def test_bbox_transform(self, snapshot: SnapshotAssertion) -> None:
        label_dict_1: dict[Hashable, float] = {"label": 1.0}
        label_dict_2: dict[Hashable, float] = {"label": 2.0}
        bboxes = [
            AxisAlignedBoundingBox(min_vertex=(1, 1), max_vertex=(2, 3)),
            AxisAlignedBoundingBox(min_vertex=(3, 2), max_vertex=(6, 8)),
        ]
        labels = [label_dict_1, label_dict_2]
        image = np.ones((30, 30, 3)).astype(np.uint8)
        inst = RandomRotationPerturber(
            limit=(180, 180),
            probability=1.0,
            seed=12,
        )
        image_out, bboxes_transformed = inst.perturb(
            image=image,
            boxes=zip(bboxes, labels, strict=False),
        )
        _, bboxes_reverted = inst.perturb(image=image_out, boxes=bboxes_transformed)

        if bboxes_reverted:
            bboxes_reverted = list(bboxes_reverted)
            assert len(bboxes_reverted) == len(bboxes)
            for bbox, label, reverted in zip(bboxes, labels, bboxes_reverted, strict=False):
                assert bbox == reverted[0]
                assert label == reverted[1]

        assert bboxes_transformed == snapshot

    def test_identity_operation(self) -> None:
        """Test that the perturber does not change the image when limit is 0."""
        image = Image.open(INPUT_IMG_FILE_PATH)
        image = np.array(image)
        inst = RandomRotationPerturber()
        out_image = perturber_assertions(perturb=inst.perturb, image=image)
        perturber_assertions(perturb=inst, image=image, expected=out_image)

    def test_consistency(self) -> None:
        """Run perturber twice with consistent seed to ensure repeatable results."""
        image = np.ones((3, 3, 3)).astype(np.uint8)

        limit = 300
        probability = 1.0
        seed = 12

        # Test perturb interface directly
        inst = RandomRotationPerturber(
            limit=limit,
            probability=probability,
            seed=seed,
        )
        out_image = perturber_assertions(perturb=inst.perturb, image=image)

        # Test callable
        perturber_assertions(
            perturb=RandomRotationPerturber(
                limit=limit,
                probability=probability,
                seed=seed,
            ),
            image=image,
            expected=out_image,
        )

    def test_non_deterministic_default(self) -> None:
        """Verify different results when seed=None (default)."""
        image = rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)
        # Use large rotation limit to ensure visible differences between non-deterministic runs
        limit = 90
        probability = 1.0

        # Create two instances with default seed=None
        inst1 = RandomRotationPerturber(limit=limit, probability=probability)
        inst2 = RandomRotationPerturber(limit=limit, probability=probability)
        out1, _ = inst1.perturb(image=image)
        out2, _ = inst2.perturb(image=image)
        # Results should (almost certainly) be different with non-deterministic default
        assert not np.array_equal(out1, out2)

    @pytest.mark.parametrize(
        ("image", "seed"),
        [
            (rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8), 2),
        ],
    )
    def test_seed_reproducibility(self, image: np.ndarray, seed: int) -> None:
        """Verify same results with explicit seed."""
        # Use large rotation limit to ensure visible differences between non-deterministic runs
        limit = 90
        probability = 1.0

        # Test perturb interface directly
        inst = RandomRotationPerturber(
            limit=limit,
            probability=probability,
            seed=seed,
        )
        out_image = perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=None,
        )
        inst = RandomRotationPerturber(  # Create new instance with same seed
            limit=limit,
            probability=probability,
            seed=seed,
        )
        perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=out_image,
        )
        inst = RandomRotationPerturber(
            limit=limit,
            probability=probability,
            seed=seed,
        )
        # Test callable
        perturber_assertions(
            perturb=inst,
            image=image,
            expected=out_image,
        )

    def test_is_static(self) -> None:
        """Verify is_static resets RNG each call."""
        image = rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)
        # Use large rotation limit to ensure visible differences between non-deterministic runs
        # is_static=True ensures identical results on repeated calls with the same input
        inst = RandomRotationPerturber(limit=90, probability=1.0, seed=42, is_static=True)
        out_image = perturber_assertions(
            perturb=inst.perturb,
            image=image,
        )
        # Second call should produce identical output due to is_static resetting RNG
        perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=out_image,
        )

    def test_is_static_warning(self) -> None:
        """Verify warning when is_static=True with seed=None."""
        with pytest.warns(UserWarning, match="is_static=True has no effect when seed=None"):
            RandomRotationPerturber(limit=90, probability=1.0, seed=None, is_static=True)

    def test_regression(self, psnr_tiff_snapshot: SnapshotAssertion) -> None:
        """Regression testing results to detect API changes."""
        grayscale_image = Image.open(INPUT_IMG_FILE_PATH)
        image = Image.new(mode="RGB", size=grayscale_image.size)
        image.paste(grayscale_image)
        image = np.array(image)
        inst = RandomRotationPerturber(
            limit=90,
            probability=1.0,
            seed=12,
        )
        out_img = perturber_assertions(
            perturb=inst.perturb,
            image=image,
        )
        psnr_tiff_snapshot.assert_match(out_img)

    @pytest.mark.parametrize(
        ("perturber", "limit", "probability", "fill", "parameters", "seed", "is_static"),
        [
            ("Rotate", 90, 1.0, [0, 0, 0], {"limit": 90, "p": 1.0, "fill": [0, 0, 0]}, 1, False),
            ("Rotate", 90, 1.0, [0, 0, 0], {"limit": 90, "p": 1.0, "fill": [0, 0, 0]}, 42, True),
            ("Rotate", 90, 1.0, [0, 0, 0], {"limit": 90, "p": 1.0, "fill": [0, 0, 0]}, None, False),
        ],
    )
    def test_configuration(
        self,
        perturber: str,
        limit: float | tuple[float, float],
        probability: float,
        fill: np.ndarray,
        parameters: dict[str, Any],
        seed: int | None,
        is_static: bool,
    ) -> None:
        """Test configuration stability."""
        inst = RandomRotationPerturber(
            limit=limit,
            probability=probability,
            seed=seed,
            is_static=is_static,
        )
        for i in configuration_test_helper(inst):
            assert i._perturber == perturber
            assert i._limit == limit
            assert i._probability == probability
            assert i._fill == fill
            assert i._parameters == parameters
            assert i.seed == seed
            assert i.is_static == is_static
