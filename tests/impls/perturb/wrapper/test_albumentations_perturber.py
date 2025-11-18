import unittest.mock as mock
from collections.abc import Hashable
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

from nrtk.impls.perturb.wrapper.albumentations_perturber import AlbumentationsPerturber
from nrtk.utils._exceptions import AlbumentationsImportError
from tests.impls import INPUT_TANK_IMG_FILE_PATH as INPUT_IMG_FILE_PATH
from tests.impls.perturb.test_perturber_utils import perturber_assertions


@pytest.mark.skipif(not AlbumentationsPerturber.is_usable(), reason=str(AlbumentationsImportError()))
class TestAlbumentationsPerturber:
    @pytest.mark.parametrize(
        ("metadata", "expectation"),
        [
            (
                {"perturber": "NotAPerturber"},
                pytest.raises(
                    ValueError,
                    match=r"Given perturber \"NotAPerturber\" is not available in Albumentations",
                ),
            ),
            (
                {"perturber": "BaseCompose"},
                pytest.raises(ValueError, match=r"Given perturber \"BaseCompose\" does not inherit \"BasicTransform\""),
            ),
            (
                {"perturber": "RandomRain", "parameters": {"NotAParam": ""}},
                pytest.warns(UserWarning, match=r"Argument\(s\) 'NotAParam' are not valid for transform RandomRain"),
            ),
            (
                {
                    "perturber": "RandomRain",
                    "parameters": {"brightness_coefficient": 0.9, "drop_width": 1, "blur_value": 5, "p": 1.0},
                },
                does_not_raise(),
            ),
            ({"perturber": "RandomFog"}, does_not_raise()),
            ({"perturber": "RandomSnow"}, does_not_raise()),
            ({"perturber": "RandomSunFlare"}, does_not_raise()),
        ],
    )
    def test_perturber_validity(
        self,
        metadata: dict[str, Any],
        expectation: AbstractContextManager,
    ) -> None:
        """Raise appropriate errors for invalid perturber or arguments."""
        if "parameters" not in metadata:
            metadata["parameters"] = dict()
        with expectation:
            AlbumentationsPerturber(perturber=metadata["perturber"], parameters=metadata["parameters"])

    @pytest.mark.parametrize(
        ("bbox"),
        [
            [0, 0, 5, 5],
            [1, 3, 5, 8],
            [8, 13, 21, 34],
        ],
    )
    def test_bbox_converters(
        self,
        bbox: list[int],
    ) -> None:
        image = np.ones((30, 30, 3)).astype(np.uint8)
        as_aab = AlbumentationsPerturber._bbox_to_aabb(bbox, image)
        as_list = AlbumentationsPerturber._aabb_to_bbox(as_aab, image)
        assert len(bbox) == len(as_list)
        assert [x == y for x, y in zip(bbox, as_list, strict=False)]

    def test_bbox_transform(self, snapshot: SnapshotAssertion) -> None:
        label_dict_1: dict[Hashable, float] = {"label": 1.0}
        label_dict_2: dict[Hashable, float] = {"label": 2.0}
        bboxes = [
            AxisAlignedBoundingBox((1, 1), (2, 3)),
            AxisAlignedBoundingBox((3, 2), (6, 8)),
        ]
        labels = [label_dict_1, label_dict_2]
        image = np.ones((30, 30, 3)).astype(np.uint8)
        inst = AlbumentationsPerturber(perturber="HorizontalFlip", parameters={"p": 1.0})
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

    @pytest.mark.parametrize(
        ("perturber", "parameters"),
        [
            ("RandomRain", {"brightness_coefficient": 0.9, "drop_width": 1, "blur_value": 5, "p": 1.0}),
            ("RandomFog", {"fog_coef_range": (0.7, 0.8), "alpha_coef": 0.1, "p": 1.0}),
            ("RandomSnow", {"snow_point_range": (0.2, 0.4), "brightness_coeff": 2.5, "p": 1.0}),
            ("RandomSunFlare", {"flare_roi": (0, 0, 1, 0.5), "angle_range": (0.25, 0.75), "p": 1.0}),
        ],
    )
    def test_consistency(
        self,
        perturber: str,
        parameters: dict[str, Any],
    ) -> None:
        """Run perturber twice with consistent seed to ensure repeatable results."""
        image = np.ones((3, 3, 3)).astype(np.uint8)

        # Test perturb interface directly
        inst = AlbumentationsPerturber(perturber=perturber, parameters=parameters, seed=1)
        out_image = perturber_assertions(perturb=inst.perturb, image=image)

        # Test callable
        perturber_assertions(
            perturb=AlbumentationsPerturber(perturber=perturber, parameters=parameters, seed=1),
            image=image,
            expected=out_image,
        )

    def test_regression(self, psnr_tiff_snapshot: SnapshotAssertion) -> None:
        """Regression testing results to detect API changes."""
        grayscale_image = Image.open(INPUT_IMG_FILE_PATH)
        image = Image.new("RGB", grayscale_image.size)
        image.paste(grayscale_image)
        image = np.array(image)
        inst = AlbumentationsPerturber(
            perturber="RandomRain",
            parameters={"brightness_coefficient": 0.9, "drop_width": 1, "blur_value": 5, "p": 1.0},
            seed=1,
        )
        out_img = perturber_assertions(
            perturb=inst.perturb,
            image=image,
        )
        psnr_tiff_snapshot.assert_match(out_img)

    @pytest.mark.parametrize(
        ("perturber", "parameters", "seed"),
        [
            ("RandomSnow", {"snow_point_range": (0.2, 0.4), "brightness_coeff": 2.5, "p": 1.0}, 1),
            ("RandomSunFlare", {"flare_roi": (0, 0, 1, 0.5), "angle_range": (0.25, 0.75), "p": 1.0}, 7),
        ],
    )
    def test_configuration(
        self,
        perturber: str,
        parameters: dict,
        seed: int,
    ) -> None:
        """Test configuration stability."""
        inst = AlbumentationsPerturber(perturber=perturber, parameters=parameters, seed=seed)
        for i in configuration_test_helper(inst):
            assert i.perturber == perturber
            assert i.parameters == parameters
            assert i.seed == seed

    def test_default_config(self) -> None:
        """Test default configuration when created with no parameters."""
        image = np.ones((3, 3, 3)).astype(np.uint8)
        inst = AlbumentationsPerturber()
        out_image = perturber_assertions(perturb=inst.perturb, image=image)

        cfg = dict()
        cfg["perturber"] = "NoOp"
        cfg["parameters"] = None
        cfg["seed"] = None
        assert (out_image == image).all()
        assert inst.get_config() == cfg


@mock.patch.object(AlbumentationsPerturber, "is_usable")
def test_missing_deps(mock_is_usable: MagicMock) -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    mock_is_usable.return_value = False
    assert not AlbumentationsPerturber.is_usable()
    with pytest.raises(AlbumentationsImportError):
        AlbumentationsPerturber("RandomRain")
