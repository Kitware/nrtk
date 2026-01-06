import unittest.mock as mock
from collections.abc import Hashable, Sequence
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image
from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from syrupy.assertion import SnapshotAssertion

from nrtk.impls.perturb_image.geometric.random_rotation_perturber import RandomRotationPerturber
from nrtk.utils._exceptions import AlbumentationsImportError
from tests.impls import INPUT_TANK_IMG_FILE_PATH as INPUT_IMG_FILE_PATH
from tests.impls.perturb_image.test_perturber_utils import perturber_assertions

rng = np.random.default_rng()


@pytest.mark.skipif(not RandomRotationPerturber.is_usable(), reason=str(AlbumentationsImportError()))
class TestRandomRotationPerturber:
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

    def test_default_seed_reproducibility(self) -> None:
        """Ensure results are reproducible with default seed (no seed parameter provided)."""
        image = rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)
        limit = 0.1
        probability = 1.0

        # Test perturb interface directly without providing seed (uses default=1)
        inst = RandomRotationPerturber(
            limit=limit,
            probability=probability,
        )
        out_image = perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=None,
        )
        inst = RandomRotationPerturber(  # Create new instance without seed
            limit=limit,
            probability=probability,
        )
        perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=out_image,
        )
        # Test callable
        inst = RandomRotationPerturber(
            limit=limit,
            probability=probability,
        )
        perturber_assertions(
            perturb=inst,
            image=image,
            expected=out_image,
        )

    @pytest.mark.parametrize(
        ("image", "seed"),
        [
            (rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8), 2),
        ],
    )
    def test_reproducibility(self, image: np.ndarray, seed: int) -> None:
        """Ensure results are reproducible when explicit seed is provided."""
        limit = 0.1
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
        ("perturber", "limit", "probability", "fill", "parameters", "seed"),
        [
            ("Rotate", 90, 1.0, [0, 0, 0], {"limit": 90, "p": 1.0, "fill": [0, 0, 0]}, 1),
        ],
    )
    def test_configuration(
        self,
        perturber: str,
        limit: float | tuple[float, float],
        probability: float,
        fill: np.ndarray,
        parameters: dict[str, Any],
        seed: int,
    ) -> None:
        """Test configuration stability."""
        inst = RandomRotationPerturber(
            limit=limit,
            probability=probability,
            seed=seed,
        )
        for i in configuration_test_helper(inst):
            assert i.perturber == perturber
            assert i.limit == limit
            assert i.probability == probability
            assert i.fill == fill
            assert i.parameters == parameters
            assert i.seed == seed


@mock.patch.object(RandomRotationPerturber, "is_usable")
def test_missing_deps(mock_is_usable: MagicMock) -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    mock_is_usable.return_value = False
    assert not RandomRotationPerturber.is_usable()
    with pytest.raises(AlbumentationsImportError):
        RandomRotationPerturber()
