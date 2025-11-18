import unittest.mock as mock
from collections.abc import Hashable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from PIL import Image
from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from syrupy.assertion import SnapshotAssertion

from nrtk.impls.perturb.geometric.random_scale_perturber import RandomScalePerturber
from nrtk.utils._exceptions import AlbumentationsImportError
from tests.impls import INPUT_TANK_IMG_FILE_PATH as INPUT_IMG_FILE_PATH
from tests.impls.perturb.test_perturber_utils import perturber_assertions


@pytest.mark.skipif(not RandomScalePerturber.is_usable(), reason=str(AlbumentationsImportError()))
class TestRandomScalePerturber:
    @pytest.mark.parametrize(
        ("limit", "interpolation", "probability", "expectation"),
        [
            (
                (-0.05, 0.05),
                cv2.INTER_LINEAR,
                -0.5,
                pytest.raises(
                    ValueError,
                    match=r"Scale probability must be between 0.0 and 1.0 inclusive.",
                ),
            ),
            (
                (-0.05, 0.05),
                cv2.INTER_LINEAR,
                1.5,
                pytest.raises(
                    ValueError,
                    match=r"Scale probability must be between 0.0 and 1.0 inclusive.",
                ),
            ),
            (
                (0.05, -0.05),
                cv2.INTER_LINEAR,
                1.0,
                pytest.raises(
                    ValueError,
                    match=r"Lower scale limit must be less than or equal to upper limit.",
                ),
            ),
            (
                -0.05,
                cv2.INTER_LINEAR,
                1.0,
                pytest.raises(
                    ValueError,
                    match=r"Lower scale limit must be less than or equal to upper limit.",
                ),
            ),
            (
                (-1.0, 1.0),
                cv2.INTER_LINEAR,
                1.0,
                pytest.raises(
                    ValueError,
                    match=r"Lower scale limit must be greater than -1.0.",
                ),
            ),
            (
                1.0,
                cv2.INTER_LINEAR,
                1.0,
                pytest.raises(
                    ValueError,
                    match=r"Lower scale limit must be greater than -1.0.",
                ),
            ),
            (
                (-0.05, 0.05),
                12345,
                1.0,
                pytest.raises(
                    ValueError,
                    match=r"Interpolation value not supported.",
                ),
            ),
            ((-0.05, 0.05), cv2.INTER_LINEAR, 1.0, does_not_raise()),
            ((-0.95, 100.0), cv2.INTER_NEAREST_EXACT, 0.7, does_not_raise()),
            (0.0, cv2.INTER_AREA, 1.0, does_not_raise()),
            (0.5, cv2.INTER_LANCZOS4, 0.0, does_not_raise()),
            ((1000.0, 1000.0), cv2.INTER_CUBIC, 0.5, does_not_raise()),
        ],
    )
    def test_perturber_validity(
        self,
        limit: float | tuple[float, float],
        interpolation: int,
        probability: float,
        expectation: AbstractContextManager,
    ) -> None:
        """Raise appropriate errors for invalid perturber arguments."""
        with expectation:
            RandomScalePerturber(
                limit=limit,
                interpolation=interpolation,
                probability=probability,
            )

    def test_bbox_transform(self, snapshot: SnapshotAssertion) -> None:
        label_dict_1: dict[Hashable, float] = {"label": 1.0}
        label_dict_2: dict[Hashable, float] = {"label": 2.0}
        bboxes = [
            AxisAlignedBoundingBox((1, 1), (2, 3)),
            AxisAlignedBoundingBox((3, 2), (6, 8)),
        ]
        labels = [label_dict_1, label_dict_2]
        image = np.ones((30, 30, 3)).astype(np.uint8)
        inst = RandomScalePerturber(
            limit=(1.0, 1.0),
            probability=1.0,
            seed=12,
        )
        image_out, bboxes_transformed = inst.perturb(
            image=image,
            boxes=zip(bboxes, labels, strict=False),
        )

        inst = RandomScalePerturber(
            limit=(-0.5, -0.5),
            probability=1.0,
            seed=12,
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
        inst = RandomScalePerturber()
        out_image = perturber_assertions(perturb=inst.perturb, image=image)
        perturber_assertions(perturb=inst, image=image, expected=out_image)

    def test_consistency(self) -> None:
        """Run perturber twice with consistent seed to ensure repeatable results."""
        image = np.ones((3, 3, 3)).astype(np.uint8)

        limit = 0.1
        interpolation = cv2.INTER_LINEAR
        probability = 1.0
        seed = 12

        # Test perturb interface directly
        inst = RandomScalePerturber(
            limit=limit,
            interpolation=interpolation,
            probability=probability,
            seed=seed,
        )
        out_image = perturber_assertions(perturb=inst.perturb, image=image)

        # Test callable
        perturber_assertions(
            perturb=RandomScalePerturber(
                limit=limit,
                interpolation=interpolation,
                probability=probability,
                seed=seed,
            ),
            image=image,
            expected=out_image,
        )

    def test_regression(self, tiff_snapshot: SnapshotAssertion) -> None:
        """Regression testing results to detect API changes."""
        grayscale_image = Image.open(INPUT_IMG_FILE_PATH)
        image = Image.new("RGB", grayscale_image.size)
        image.paste(grayscale_image)
        image = np.array(image)
        inst = RandomScalePerturber(
            limit=(-0.1, 0.2),
            interpolation=cv2.INTER_LINEAR_EXACT,
            probability=1.0,
            seed=12,
        )
        out_img = perturber_assertions(
            perturb=inst.perturb,
            image=image,
        )
        tiff_snapshot.assert_match(out_img)

    @pytest.mark.parametrize(
        ("perturber", "limit", "interpolation", "probability", "parameters", "seed"),
        [
            (
                "RandomScale",
                (-0.1, 0.1),
                1.0,
                cv2.INTER_LINEAR,
                {"scale_limit": (-0.1, 0.1), "interpolation": cv2.INTER_LINEAR, "p": 1.0},
                1,
            ),
        ],
    )
    def test_configuration(
        self,
        perturber: str,
        limit: float | tuple[float, float],
        interpolation: int,
        probability: float,
        parameters: dict[str, Any],
        seed: int | None,
    ) -> None:
        """Test configuration stability."""
        inst = RandomScalePerturber(
            limit=limit,
            probability=probability,
            seed=seed,
        )
        for i in configuration_test_helper(inst):
            assert i.perturber == perturber
            assert i.limit == limit
            assert i.interpolation == interpolation
            assert i.probability == probability
            assert i.parameters == parameters
            assert i.seed == seed


@mock.patch.object(RandomScalePerturber, "is_usable")
def test_missing_deps(mock_is_usable: MagicMock) -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    mock_is_usable.return_value = False
    assert not RandomScalePerturber.is_usable()
    with pytest.raises(AlbumentationsImportError):
        RandomScalePerturber()
