from __future__ import annotations

import unittest.mock as mock
from collections.abc import Hashable, Iterable, Sequence
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image
from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from syrupy.assertion import SnapshotAssertion

from nrtk.impls.perturb_image.generic.water_droplet_perturber import WaterDropletPerturber
from nrtk.utils._exceptions import WaterDropletImportError
from tests.impls import INPUT_TANK_IMG_FILE_PATH as INPUT_IMG_FILE_PATH
from tests.impls.perturb_image.test_perturber_utils import perturber_assertions
from tests.impls.test_pybsm_utils import TIFFImageSnapshotExtension


@pytest.fixture
def tiff_snapshot(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(TIFFImageSnapshotExtension)


@pytest.mark.skipif(not WaterDropletPerturber.is_usable(), reason=str(WaterDropletImportError()))
class TestWaterDropletPerturber:
    def test_default_consistency(
        self,
    ) -> None:
        """Run on a dummy image to ensure multiple calls produce the same result."""
        img = np.ones((3, 3, 3)).astype(np.uint8)

        inst = WaterDropletPerturber()

        out_img = perturber_assertions(perturb=inst, image=img, expected=None, additional_params=None)

        perturber_assertions(perturb=inst, image=img, expected=out_img, additional_params=None)

    @pytest.mark.parametrize(
        (
            "size_range",
            "num_drops",
            "blur_strength",
            "psi",
            "n_air",
            "n_water",
            "f_x",
            "f_y",
            "seed",
        ),
        [
            ((0.0, 1.0), 20, 0.25, 90.0 / 180.0 * np.pi, 1.0, 1.33, 400, 400, 0),
            ((0.0, 1.0), 25, 0.25, 90.0 / 180.0 * np.pi, 1.0, 1.33, 400, 400, 0),
            ((0.0, 1.0), 50, 0.25, 90.0 / 180.0 * np.pi, 1.0, 1.33, 400, 400, 0),
            ((0.0, 1.0), 100, 0.25, 90.0 / 180.0 * np.pi, 1.0, 1.33, 400, 400, 0),
            ((0.0, 1.0), 20, 0.5, 90.0 / 180.0 * np.pi, 1.0, 1.33, 400, 400, 0),
            ((0.0, 1.0), 20, 0.75, 90.0 / 180.0 * np.pi, 1.0, 1.33, 400, 400, 0),
            ((0.0, 1.0), 20, 1.0, 90.0 / 180.0 * np.pi, 1.0, 1.33, 400, 400, 0),
        ],
    )
    def test_consistency(
        self,
        size_range: Sequence[float],
        num_drops: int,
        blur_strength: float,
        psi: float,
        n_air: float,
        n_water: float,
        f_x: int,
        f_y: int,
        seed: int | None,
    ) -> None:
        """Run on a dummy image to ensure multiple calls produce the same result."""
        img = np.ones((3, 3, 3)).astype(np.uint8)

        inst = WaterDropletPerturber(
            size_range=size_range,
            num_drops=num_drops,
            blur_strength=blur_strength,
            psi=psi,
            n_air=n_air,
            n_water=n_water,
            f_x=f_x,
            f_y=f_y,
            seed=seed,
        )

        out_img = perturber_assertions(perturb=inst, image=img, expected=None, additional_params=None)

        perturber_assertions(perturb=inst, image=img, expected=out_img, additional_params=None)

    @pytest.mark.parametrize(
        (
            "size_range",
            "num_drops",
            "blur_strength",
            "psi",
            "n_air",
            "n_water",
            "f_x",
            "f_y",
            "seed",
        ),
        [
            ((0.0, 1.0), 20, 0.25, 90.0 / 180.0 * np.pi, 1.0, 1.33, 400, 400, 0),
            ((0.0, 1.0), 20, 0.25, 90.0 / 180.0 * np.pi, 1.0, 1.33, 400, 400, 12),
            ((0.0, 1.0), 20, 0.25, 90.0 / 180.0 * np.pi, 1.0, 1.33, 400, 400, 234),
        ],
    )
    def test_regression(
        self,
        tiff_snapshot: SnapshotAssertion,
        size_range: Sequence[float],
        num_drops: int,
        blur_strength: float,
        psi: float,
        n_air: float,
        n_water: float,
        f_x: int,
        f_y: int,
        seed: int | None,
    ) -> None:
        """Regression testing results to detect API changes."""
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        inst = WaterDropletPerturber(
            size_range=size_range,
            num_drops=num_drops,
            blur_strength=blur_strength,
            psi=psi,
            n_air=n_air,
            n_water=n_water,
            f_x=f_x,
            f_y=f_y,
            seed=seed,
        )
        out_img = perturber_assertions(
            perturb=inst.perturb,
            image=image,
        )
        tiff_snapshot.assert_match(out_img)

    @pytest.mark.parametrize(
        (
            "size_range",
            "num_drops",
            "blur_strength",
            "psi",
            "n_air",
            "n_water",
            "f_x",
            "f_y",
            "seed",
        ),
        [
            ((0.0, 1.0), 20, 0.25, 90.0 / 180.0 * np.pi, 1.0, 1.33, 400, 400, 0),
        ],
    )
    def test_configuration(
        self,
        size_range: Sequence[float],
        num_drops: int,
        blur_strength: float,
        psi: float,
        n_air: float,
        n_water: float,
        f_x: int,
        f_y: int,
        seed: int,
    ) -> None:
        """Test configuration stability."""
        inst = WaterDropletPerturber(
            size_range=size_range,
            num_drops=num_drops,
            blur_strength=blur_strength,
            psi=psi,
            n_air=n_air,
            n_water=n_water,
            f_x=f_x,
            f_y=f_y,
            seed=seed,
        )
        for i in configuration_test_helper(inst):
            assert i.size_range == size_range
            assert i.num_drops == num_drops
            assert i.blur_strength == blur_strength
            assert i.psi == psi
            assert i.n_air == n_air
            assert i.n_water == n_water
            assert i.f_x == f_x
            assert i.f_y == f_y
            assert i.seed == seed

    @pytest.mark.parametrize(
        "boxes",
        [
            None,
            [(AxisAlignedBoundingBox((0, 0), (1, 1)), {"test": 0.0})],
            [
                (AxisAlignedBoundingBox((0, 0), (1, 1)), {"test": 0.0}),
                (AxisAlignedBoundingBox((2, 2), (3, 3)), {"test2": 1.0}),
            ],
        ],
    )
    def test_perturb_with_boxes(self, boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]) -> None:
        """Test that bounding boxes do not change during perturb."""
        inst = WaterDropletPerturber()
        _, out_boxes = inst.perturb(np.ones((256, 256, 3)), boxes=boxes)
        assert boxes == out_boxes


@mock.patch.object(WaterDropletPerturber, "is_usable")
def test_missing_deps(mock_is_usable: MagicMock) -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    mock_is_usable.return_value = False
    assert not WaterDropletPerturber.is_usable()
    with pytest.raises(WaterDropletImportError):
        WaterDropletPerturber()
