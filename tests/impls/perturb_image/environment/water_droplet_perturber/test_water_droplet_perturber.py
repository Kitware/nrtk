from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence

import numpy as np
import pytest
from PIL import Image
from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from syrupy.assertion import SnapshotAssertion

from nrtk.impls.perturb_image.environment import WaterDropletPerturber
from nrtk.impls.perturb_image.environment._water_droplet_perturber import (
    compute_refraction_mapping_impl,
    points_in_polygon_impl,
)
from tests.impls import INPUT_TANK_IMG_FILE_PATH as INPUT_IMG_FILE_PATH
from tests.impls.perturb_image.perturber_tests_mixin import PerturberTestsMixin
from tests.impls.perturb_image.perturber_utils import perturber_assertions

rng = np.random.default_rng(2345)
reproduce_rng = np.random.default_rng(23456)


@pytest.mark.waterdroplet
class TestWaterDropletPerturber(PerturberTestsMixin):
    impl_class = WaterDropletPerturber

    def test_non_deterministic_default(self) -> None:
        """Verify different results when seed=None (default)."""
        image = reproduce_rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)

        # Create two instances with default seed=None
        inst1 = WaterDropletPerturber()
        inst2 = WaterDropletPerturber()
        out1, _ = inst1.perturb(image=image)
        out2, _ = inst2.perturb(image=image)
        # Results should (almost certainly) be different with non-deterministic default
        assert not np.array_equal(out1, out2)

    @pytest.mark.parametrize(
        ("image", "seed"),
        [
            (reproduce_rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8), 2),
        ],
    )
    def test_seed_reproducibility(self, image: np.ndarray, seed: int) -> None:
        """Verify same results with explicit seed."""
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

    def test_is_static(self) -> None:
        """Verify is_static resets RNG each call."""
        image = reproduce_rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)
        inst = WaterDropletPerturber(seed=42, is_static=True)
        out1 = perturber_assertions(perturb=inst.perturb, image=image, expected=None)
        # Same result each call with is_static
        perturber_assertions(perturb=inst.perturb, image=image, expected=out1)

    def test_is_static_warning(self) -> None:
        """Verify warning when is_static=True with seed=None."""
        with pytest.warns(UserWarning, match="is_static=True has no effect"):
            WaterDropletPerturber(seed=None, is_static=True)

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
        """Run on a dummy image to ensure multiple calls produce the same result with is_static."""
        img = np.ones((3, 3, 3)).astype(np.uint8)

        # Use is_static=True to ensure consistent results across multiple calls
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
            is_static=True,
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
            ((0.0, 1.0), 20, 0.25, 90.0 / 180.0 * np.pi, 1.0, 1.33, 400, 400, 42),
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
        image = np.array(Image.open(INPUT_IMG_FILE_PATH).resize((128, 128)))
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
            "is_static",
        ),
        [
            ((0.0, 1.0), 20, 0.25, 90.0 / 180.0 * np.pi, 1.0, 1.33, 400, 400, 0, False),
            ((0.0, 1.0), 20, 0.25, 90.0 / 180.0 * np.pi, 1.0, 1.33, 400, 400, 42, True),
            ((0.0, 1.0), 20, 0.25, 90.0 / 180.0 * np.pi, 1.0, 1.33, 400, 400, None, False),
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
        seed: int | None,
        is_static: bool,
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
            is_static=is_static,
        )
        for i in configuration_test_helper(inst):
            assert list(i.size_range) == list(size_range)
            assert i.num_drops == num_drops
            assert i.blur_strength == blur_strength
            assert i.psi == psi
            assert i.n_air == n_air
            assert i.n_water == n_water
            assert i.f_x == f_x
            assert i.f_y == f_y
            assert i.seed == seed
            assert i.is_static == is_static

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


@pytest.mark.waterdroplet
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


@pytest.mark.waterdroplet
@pytest.mark.parametrize(
    ("points", "expected"),
    [
        ([[0.5, 0.5], [0.25, 0.25], [0.75, 0.75]], [True, True, True]),  # all inside
        ([[2.0, 2.0], [-0.5, 0.5], [0.5, -0.5]], [False, False, False]),  # all outside
        ([[0.5, 0.5], [2.0, 2.0], [0.25, 0.75]], [True, False, True]),  # mixed
    ],
)
def test_points_in_polygon_impl(points: list[list[float]], expected: list[bool]) -> None:
    """Test points_in_polygon_impl with various point configurations."""
    polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    points_arr = np.array(points, dtype=np.float64)
    result = points_in_polygon_impl(points=points_arr, polygon=polygon)
    np.testing.assert_array_equal(result, np.array(expected))


@pytest.mark.waterdroplet
def test_compute_refraction_mapping_impl() -> None:
    """Test compute_refraction_mapping_impl produces valid output."""
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

    u_out, v_out = compute_refraction_mapping_impl(
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
