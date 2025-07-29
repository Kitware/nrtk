from __future__ import annotations

import unittest.mock as mock
from unittest.mock import MagicMock

import numpy as np
import pytest
from syrupy.assertion import SnapshotAssertion

from nrtk.impls.perturb_image.generic.utils.water_droplet_perturber_utils import (
    ccw_sort,
    get_bezier_curve,
    get_random_points_within_min_dist,
)
from nrtk.impls.perturb_image.generic.water_droplet_perturber import WaterDropletPerturber
from nrtk.utils._exceptions import WaterDropletImportError
from tests.impls.test_pybsm_utils import TIFFImageSnapshotExtension

rng = np.random.default_rng(2345)


@pytest.mark.skipif(not WaterDropletPerturber.is_usable(), reason=str(WaterDropletImportError()))
class TestWaterDropletPerturberUtils:
    def test_ccw_sort(
        self,
        snapshot: SnapshotAssertion,
    ) -> None:
        """Regression testing for the `ccw_sort` util function."""
        x_in = np.linspace(0, 100, 50)
        x_out, y_out = np.meshgrid(x_in, x_in)
        points = np.vstack((x_out.ravel(), y_out.ravel())).T
        points = ccw_sort(points=points)

        assert TIFFImageSnapshotExtension.ndarray2bytes(points) == snapshot(extension_class=TIFFImageSnapshotExtension)

    @pytest.mark.parametrize(
        ("rad", "edgy"),
        [(0.2, 0), (0.5, 0), (0.7, 0), (0.2, 1), (0.5, 1), (0.7, 1)],
    )
    def test_regression_get_bezier_curve(
        self,
        snapshot: SnapshotAssertion,
        rad: float,
        edgy: float,
    ) -> None:
        """Regression testing for the `get_bezier_curve` util function."""
        x_in = np.linspace(0, 100, 50)
        x_out, y_out = np.meshgrid(x_in, x_in)
        points = np.vstack((x_out.ravel(), y_out.ravel())).T
        x, y = get_bezier_curve(points=points, rad=rad, edgy=edgy)
        curve_points = np.column_stack((x, y))
        assert TIFFImageSnapshotExtension.ndarray2bytes(curve_points) == snapshot(
            extension_class=TIFFImageSnapshotExtension,
        )

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
        snapshot: SnapshotAssertion,
        n: int,
        scale: float,
        min_dst: float | None,
        recursive: int,
    ) -> None:
        """Regression testing for the `get_random_points_within_min_dist` util function."""
        rand_points = get_random_points_within_min_dist(
            rng=rng,
            n=n,
            scale=scale,
            min_dst=min_dst,
            recursive=recursive,
        )
        assert TIFFImageSnapshotExtension.ndarray2bytes(rand_points) == snapshot(
            extension_class=TIFFImageSnapshotExtension,
        )


@mock.patch.object(WaterDropletPerturber, "is_usable")
def test_missing_deps(mock_is_usable: MagicMock) -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    mock_is_usable.return_value = False
    assert not WaterDropletPerturber.is_usable()
    with pytest.raises(WaterDropletImportError):
        WaterDropletPerturber()
