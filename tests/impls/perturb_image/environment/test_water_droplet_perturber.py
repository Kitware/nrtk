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

from nrtk.impls.perturb_image.environment.water_droplet_perturber import (
    WaterDropletPerturber,
    _compute_refraction_mapping_impl,
    _points_in_polygon_impl,
)
from nrtk.utils._exceptions import WaterDropletImportError
from tests.impls import INPUT_TANK_IMG_FILE_PATH as INPUT_IMG_FILE_PATH
from tests.impls.perturb_image.test_perturber_utils import perturber_assertions

rng = np.random.default_rng(2345)
reproduce_rng = np.random.default_rng(23456)


@pytest.mark.skipif(not WaterDropletPerturber.is_usable(), reason=str(WaterDropletImportError()))
class TestWaterDropletPerturber:
    def test_default_seed_reproducibility(self) -> None:
        """Ensure results are reproducible with default seed (no seed parameter provided)."""
        image = reproduce_rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)

        # Test perturb interface directly without providing seed (uses default=1)
        inst = WaterDropletPerturber()
        out_image = perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=None,
        )

        # Create new instance without seed
        inst = WaterDropletPerturber()
        perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=out_image,
        )

        # Test callable
        inst = WaterDropletPerturber()
        perturber_assertions(
            perturb=inst,
            image=image,
            expected=out_image,
        )

    @pytest.mark.parametrize(
        ("image", "seed"),
        [
            (reproduce_rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8), 2),
        ],
    )
    def test_reproducibility(self, image: np.ndarray, seed: int) -> None:
        """Ensure results are reproducible when explicit seed is provided."""
        # Test perturb interface directly
        inst = WaterDropletPerturber(seed=seed)
        out_image = perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=None,
        )

        # Create new instance with same seed
        inst = WaterDropletPerturber(seed=seed)
        perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=out_image,
        )

        # Test callable
        inst = WaterDropletPerturber(seed=seed)
        perturber_assertions(
            perturb=inst,
            image=image,
            expected=out_image,
        )

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

        out_img = perturber_assertions(perturb=inst, image=img, expected=None)

        perturber_assertions(perturb=inst, image=img, expected=out_img)

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
        psnr_tiff_snapshot: SnapshotAssertion,
        ssim_tiff_snapshot: SnapshotAssertion,
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
        psnr_tiff_snapshot.assert_match(out_img)
        ssim_tiff_snapshot.assert_match(out_img)

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
            [(AxisAlignedBoundingBox(min_vertex=(0, 0), max_vertex=(1, 1)), {"test": 0.0})],
            [
                (AxisAlignedBoundingBox(min_vertex=(0, 0), max_vertex=(1, 1)), {"test": 0.0}),
                (AxisAlignedBoundingBox(min_vertex=(2, 2), max_vertex=(3, 3)), {"test2": 1.0}),
            ],
        ],
    )
    def test_perturb_with_boxes(self, boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]) -> None:
        """Test that bounding boxes do not change during perturb."""
        inst = WaterDropletPerturber()
        _, out_boxes = inst.perturb(image=np.ones((256, 256, 3)), boxes=boxes)
        assert boxes == out_boxes

    @pytest.mark.parametrize("num_drops", [0, 1])
    def test_few_droplets_no_overlap_check(self, num_drops: int) -> None:
        """Test that perturb handles 0 or 1 droplets without overlap removal issues."""
        # Use a local RNG to avoid affecting other tests that use the module-level rng
        _rng = np.random.default_rng(2345)
        image = _rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)
        inst = WaterDropletPerturber(num_drops=num_drops, seed=42)
        out_image, _ = inst.perturb(image=image)
        # Verify the output is valid (same shape, uint8 type)
        assert out_image.shape == image.shape
        assert out_image.dtype == np.uint8


@pytest.mark.skipif(not WaterDropletPerturber.is_usable(), reason=str(WaterDropletImportError()))
class TestWaterDropletPerturberUtils:
    def test_ccw_sort(
        self,
        fuzzy_snapshot: SnapshotAssertion,
    ) -> None:
        """Regression testing for the `ccw_sort` util function."""
        x_in = np.linspace(0, 100, 50)
        x_out, y_out = np.meshgrid(x_in, x_in)
        points = np.vstack((x_out.ravel(), y_out.ravel())).T
        points = WaterDropletPerturber.ccw_sort(points=points)

        fuzzy_snapshot.assert_match(points)

    @pytest.mark.parametrize(
        ("rad", "edgy"),
        [(0.2, 0), (0.5, 0), (0.7, 0), (0.2, 1), (0.5, 1), (0.7, 1)],
    )
    def test_regression_get_bezier_curve(
        self,
        fuzzy_snapshot: SnapshotAssertion,
        rad: float,
        edgy: float,
    ) -> None:
        """Regression testing for the `get_bezier_curve` util function."""
        x_in = np.linspace(0, 100, 50)
        x_out, y_out = np.meshgrid(x_in, x_in)
        points = np.vstack((x_out.ravel(), y_out.ravel())).T
        x, y = WaterDropletPerturber.get_bezier_curve(points=points, rad=rad, edgy=edgy)
        curve_points = np.column_stack((x, y))
        fuzzy_snapshot.assert_match(curve_points)

    @pytest.mark.parametrize(
        ("n", "scale", "min_dst", "recursive"),
        [
            (5, 0.8, 0.2, 0),
            (5, 0.8, 0.2, 50),
            (100, 0.8, 0.2, 0),
            (100, 0.8, 0.2, 50),
        ],
    )
    def test_regression_get_random_points_within_min_dist(
        self,
        fuzzy_snapshot: SnapshotAssertion,
        n: int,
        scale: float,
        min_dst: float | None,
        recursive: int,
    ) -> None:
        """Regression testing for the `get_random_points_within_min_dist` util function."""
        _rng = np.random.default_rng(2345)
        rand_points = WaterDropletPerturber.get_random_points_within_min_dist(
            rng=_rng,
            n=n,
            scale=scale,
            min_dst=min_dst,
            recursive=recursive,
        )
        fuzzy_snapshot.assert_match(rand_points)


@mock.patch.object(WaterDropletPerturber, "is_usable")
def test_missing_deps(mock_is_usable: MagicMock) -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    mock_is_usable.return_value = False
    assert not WaterDropletPerturber.is_usable()
    with pytest.raises(WaterDropletImportError):
        WaterDropletPerturber()


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        ([[0.5, 0.5], [0.25, 0.25], [0.75, 0.75]], [True, True, True]),  # all inside
        ([[2.0, 2.0], [-0.5, 0.5], [0.5, -0.5]], [False, False, False]),  # all outside
        ([[0.5, 0.5], [2.0, 2.0], [0.25, 0.75]], [True, False, True]),  # mixed
    ],
)
def test_points_in_polygon_impl(points: list[list[float]], expected: list[bool]) -> None:
    """Test _points_in_polygon_impl with various point configurations."""
    polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    points_arr = np.array(points, dtype=np.float64)
    result = _points_in_polygon_impl(points=points_arr, polygon=polygon)
    np.testing.assert_array_equal(result, np.array(expected))


def test_compute_refraction_mapping_impl() -> None:
    """Test _compute_refraction_mapping_impl produces valid output."""
    # Create minimal test inputs
    xs = np.array([5, 6], dtype=np.int64)
    ys = np.array([5, 6], dtype=np.int64)

    # Create a simple glass coordinate system (10x10x3)
    gls = np.zeros((10, 10, 3), dtype=np.float64)
    for i in range(10):
        for j in range(10):
            gls[i, j] = [float(i), float(j), 30.0]

    normal = np.array([0.0, -1.0, 1.0], dtype=np.float64)
    n_air = 1.0
    n_water = 1.33
    m_dist = 30
    b_dist = 1000
    center = np.array([5.0, 5.0, 29.5], dtype=np.float64)
    radius = 0.5
    intrinsic = np.array([[400, 0, 5], [0, 400, 5], [0, 0, 1]], dtype=np.float64)

    u_out, v_out = _compute_refraction_mapping_impl(
        xs=xs,
        ys=ys,
        gls=gls,
        normal=normal,
        n_air=n_air,
        n_water=n_water,
        M=m_dist,
        B=b_dist,
        center=center,
        radius=radius,
        intrinsic=intrinsic,
    )

    # Verify output shape and type
    assert u_out.shape == (2,)
    assert v_out.shape == (2,)
    assert u_out.dtype == np.float64
    assert v_out.dtype == np.float64
