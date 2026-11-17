"""Tests for the jitter render helpers (PSF, convolution, warp, box bounds)."""

from __future__ import annotations

import numpy as np
import pytest

from nrtk.impls.perturb_video.optical._jitter import _render


@pytest.mark.scipy
def test_smear_psf_horizontal_path_is_horizontal_kernel() -> None:
    """A horizontal path produces a normalized kernel that is wider than it is tall."""
    path_x = np.linspace(-3.0, 3.0, 64)  # 6 px horizontal excursion, finely sampled
    path_y = np.zeros_like(path_x)
    psf = _render.directional_smear_psf(path_px_x=path_x, path_px_y=path_y)
    assert psf is not None
    assert psf.sum() == pytest.approx(1.0, abs=1e-9)
    rows = np.flatnonzero(psf.sum(axis=1) > 1e-9)
    cols = np.flatnonzero(psf.sum(axis=0) > 1e-9)
    assert cols.size > rows.size  # wider than tall


@pytest.mark.scipy
def test_smear_psf_preserves_centroid_of_mean_subtracted_path() -> None:
    """A mean-subtracted path yields a kernel centered on the kernel midpoint.

    Bilinear splatting preserves the first moment; border-clipping may move it by <= ~0.05 px.
    """
    rng = np.random.default_rng(0)
    px = rng.normal(loc=0.0, scale=2.0, size=256)
    py = rng.normal(loc=0.0, scale=2.0, size=256)
    px -= px.mean()
    py -= py.mean()
    psf = _render.directional_smear_psf(path_px_x=px, path_px_y=py)
    assert psf is not None
    ys, xs = np.mgrid[0 : psf.shape[0], 0 : psf.shape[1]]
    center = (psf.shape[0] - 1) / 2.0
    assert float((psf * xs).sum()) == pytest.approx(center, abs=0.06)
    assert float((psf * ys).sum()) == pytest.approx(center, abs=0.06)


@pytest.mark.scipy
def test_smear_psf_full_cycle_is_a_streak_not_spikes() -> None:
    """A finely sampled full-cycle path yields a contiguous streak, not isolated spikes.

    At ~4 samples/cycle a one-cycle 3 px sinusoid rasterizes to isolated spikes; on a fine
    grid the kernel must have contiguous support.
    """
    t = np.arange(64) / 64.0
    px = 3.0 * np.sin(2.0 * np.pi * t)
    px -= px.mean()
    psf = _render.directional_smear_psf(path_px_x=px, path_px_y=np.zeros_like(px))
    assert psf is not None
    center_row = psf[(psf.shape[0] - 1) // 2]
    assert np.flatnonzero(center_row > 1e-6).size >= 5  # contiguous support, not 3 dots


@pytest.mark.scipy
def test_smear_psf_subpixel_returns_none() -> None:
    """Sub-pixel excursions produce no kernel (no meaningful smear)."""
    assert _render.directional_smear_psf(path_px_x=np.array([0.0, 0.1]), path_px_y=np.array([0.0, 0.05])) is None


@pytest.mark.scipy
def test_convolve_preserves_dtype_and_range() -> None:
    """Convolution preserves dtype and never wraps out-of-range values."""
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[16, 16] = 255
    psf = np.full((3, 3), 1.0 / 9.0)
    out = _render.convolve_psf(image=img, psf=psf)
    assert out.dtype == np.uint8
    assert int(out.max()) <= 255  # rint+clip cast, never wraparound
    assert int(out.min()) >= 0


@pytest.mark.scipy
def test_warp_translation_moves_dot() -> None:
    """Pure translation moves content by exactly (dx, dy)."""
    img = np.zeros((101, 101), dtype=np.float64)
    img[50, 50] = 1.0
    h = np.full(101, 0.0)
    out = _render.warp_frame(
        image=img,
        dx_rows=h + 5.0,
        dy_rows=h - 3.0,
        roll_rad=0.0,
        ifov=1e-6,
        order=1,
        fill=np.zeros(1),
    )
    assert np.unravel_index(int(np.argmax(out)), out.shape) == (47, 55)  # +5 col, -3 row


@pytest.mark.scipy
def test_warp_pure_roll_convention_ccw_on_screen() -> None:
    """Positive roll is counter-clockwise as displayed: a dot right of center moves up."""
    img = np.zeros((101, 101), dtype=np.float64)
    img[50, 80] = 1.0
    z = np.zeros(101)
    out = _render.warp_frame(image=img, dx_rows=z, dy_rows=z, roll_rad=np.pi / 2, ifov=1e-6, order=1, fill=np.zeros(1))
    assert np.unravel_index(int(np.argmax(out)), out.shape) == (20, 50)  # (row 50, col 80) -> (row 20, col 50)


def _zero_motion_coords(*, height: int, width: int) -> np.ndarray:
    z = np.zeros(height)
    return _render.source_coords(
        dx_rows=z,
        dy_rows=z,
        roll_rad=0.0,
        ifov=1e-4,
        height=height,
        width=width,
        border=True,
    )


@pytest.mark.scipy
@pytest.mark.parametrize(
    ("box", "expected"),
    [
        ((2.0, 3.0, 7.0, 9.0), (2.0, 3.0, 7.0, 9.0)),
        ((1.3, 2.2, 5.6, 6.7), (1.3, 2.2, 5.6, 6.7)),  # fractional edges are kept
        ((0.0, 0.0, 16.0, 12.0), (0.0, 0.0, 16.0, 12.0)),  # full frame
        ((15.0, 2.0, 16.0, 5.0), (15.0, 2.0, 16.0, 5.0)),  # last column
        ((15.6, 2.0, 16.0, 5.0), (15.6, 2.0, 16.0, 5.0)),  # thinner than a pixel
        ((1.6, 1.6, 1.9, 1.9), (1.6, 1.6, 1.9, 1.9)),
        ((-2.0, -2.0, 3.0, 3.0), (0.0, 0.0, 3.0, 3.0)),  # clipped to the frame
        ((10.0, 8.0, 20.0, 15.0), (10.0, 8.0, 16.0, 12.0)),
    ],
)
def test_box_bounds_zero_motion(
    box: tuple[float, float, float, float],
    expected: tuple[float, float, float, float],
) -> None:
    """With no motion a box comes back unchanged, clipped to the frame."""
    coords = _zero_motion_coords(height=12, width=16)
    assert _render.box_bounds(coords=coords, boxes=[box])[0] == pytest.approx(expected)


@pytest.mark.scipy
@pytest.mark.parametrize(
    "box",
    [(3.0, 3.0, 3.0, 6.0), (-3.0, 2.0, -0.4, 5.0), (16.5, 2.0, 20.0, 5.0), (2.0, 12.0, 5.0, 14.0)],
)
def test_box_bounds_empty(box: tuple[float, float, float, float]) -> None:
    """Zero-area boxes and boxes outside the frame are reported as None."""
    coords = _zero_motion_coords(height=12, width=16)
    assert _render.box_bounds(coords=coords, boxes=[box]) == [None]


@pytest.mark.scipy
def test_box_bounds_subpixel_translation() -> None:
    """A global shift moves the box by exactly that shift."""
    z = np.zeros(40)
    coords = _render.source_coords(
        dx_rows=z + 2.3,
        dy_rows=z - 1.6,
        roll_rad=0.0,
        ifov=0.0,
        height=40,
        width=50,
        border=True,
    )
    bounds = _render.box_bounds(coords=coords, boxes=[(10.25, 12.5, 20.0, 30.75), (45.0, 0.0, 50.0, 3.0)])
    assert bounds[0] == pytest.approx((12.55, 10.9, 22.3, 29.15))
    assert bounds[1] == pytest.approx((47.3, 0.0, 50.0, 1.4))


@pytest.mark.scipy
def test_box_bounds_partly_out_of_frame() -> None:
    """Only the part of a box that stays in the frame counts toward its extent."""
    h = 40
    dx_rows = -0.05 * np.arange(h)  # the left column slides out of frame row by row
    coords = _render.source_coords(
        dx_rows=dx_rows,
        dy_rows=np.zeros(h),
        roll_rad=0.0,
        ifov=0.0,
        height=h,
        width=20,
        border=True,
    )
    bounds = _render.box_bounds(coords=coords, boxes=[(0.0, 0.0, 1.0, 40.0)])[0]
    assert bounds == pytest.approx((0.0, 0.0, 1.0, 20.5))  # row 20 has moved a full pixel


@pytest.mark.scipy
def test_box_bounds_shifted_out_of_frame() -> None:
    """A box whose content is shifted entirely out of frame is reported as None."""
    z = np.zeros(10)
    coords = _render.source_coords(
        dx_rows=z + 50.0,
        dy_rows=z,
        roll_rad=0.0,
        ifov=1e-6,
        height=10,
        width=10,
        border=True,
    )
    assert _render.box_bounds(coords=coords, boxes=[(0.0, 0.0, 10.0, 10.0)]) == [None]


@pytest.mark.scipy
def test_warp_and_box_bounds_agree() -> None:
    """The box of a single pixel lands on that pixel in the warped image.

    Uses roll, varying per-row displacement and keystone together, so a sign or handedness
    mismatch between the image warp and the box map shows up.
    """
    img = np.zeros((101, 101), dtype=np.float64)
    img[30, 90] = 1.0
    rows = np.arange(101, dtype=np.float64)
    dx_rows = 4.0 + 2.0 * np.sin(2.0 * np.pi * rows / 101.0)  # rolling-shutter-like field
    dy_rows = -2.0 + 1.0 * np.cos(2.0 * np.pi * rows / 101.0)
    out = _render.warp_frame(
        image=img,
        dx_rows=dx_rows,
        dy_rows=dy_rows,
        roll_rad=0.05,
        ifov=2e-3,
        order=1,
        fill=np.zeros(1),
    )
    coords = _render.source_coords(
        dx_rows=dx_rows,
        dy_rows=dy_rows,
        roll_rad=0.05,
        ifov=2e-3,
        height=101,
        width=101,
        border=True,
    )
    bounds = _render.box_bounds(coords=coords, boxes=[(90.0, 30.0, 91.0, 31.0)])[0]
    got_row, got_col = np.unravel_index(int(np.argmax(out)), out.shape)
    assert bounds is not None
    x1, y1, x2, y2 = bounds
    assert x1 <= got_col + 0.5 <= x2
    assert y1 <= got_row + 0.5 <= y2
    assert x2 - x1 <= 1.5  # a one-pixel box stays about one pixel
    assert y2 - y1 <= 1.5


@pytest.mark.scipy
def test_box_bounds_row_varying_shift() -> None:
    """Under row-varying vertical displacement the box lands on the output row that reads the object.

    ``warp_frame`` fills output row r from source row r - dy_rows[r], so the object does not
    move by the displacement of its own source row.
    """
    h = 101
    rows = np.arange(h, dtype=np.float64)
    dy_rows = 30.0 * np.sin(2.0 * np.pi * rows / h) ** 2  # up to ~0.9 px/row
    z = np.zeros(h)
    img = np.zeros((h, h))
    img[5, 50] = 1.0
    out = _render.warp_frame(image=img, dx_rows=z, dy_rows=dy_rows, roll_rad=0.0, ifov=1e-6, order=1, fill=np.zeros(1))
    coords = _render.source_coords(
        dx_rows=z,
        dy_rows=dy_rows,
        roll_rad=0.0,
        ifov=1e-6,
        height=h,
        width=h,
        border=True,
    )
    bounds = _render.box_bounds(coords=coords, boxes=[(50.0, 5.0, 51.0, 6.0)])[0]
    got_row = np.unravel_index(int(np.argmax(out)), out.shape)[0]
    assert bounds is not None
    assert bounds[1] <= got_row + 0.5 <= bounds[3]
    assert abs((5.0 + dy_rows[5]) - got_row) > 1.5


@pytest.mark.scipy
def test_box_bounds_folded_readout() -> None:
    """When the readout folds, the box spans every output row that reads the object.

    With more than a pixel of displacement change per row, ``r - dy_rows[r]`` is not
    monotonic and the same source row is read by several output rows.
    """
    h = 101
    rows = np.arange(h, dtype=np.float64)
    dy_rows = -30.0 * np.sin(2.0 * np.pi * rows / h)  # up to 1.87 px/row
    z = np.zeros(h)
    img = np.zeros((h, h))
    img[50, 50] = 1.0
    out = _render.warp_frame(image=img, dx_rows=z, dy_rows=dy_rows, roll_rad=0.0, ifov=1e-6, order=1, fill=np.zeros(1))
    coords = _render.source_coords(
        dx_rows=z,
        dy_rows=dy_rows,
        roll_rad=0.0,
        ifov=1e-6,
        height=h,
        width=h,
        border=True,
    )
    bounds = _render.box_bounds(coords=coords, boxes=[(50.0, 50.0, 51.0, 51.0)])[0]
    lit_rows = np.flatnonzero(out.max(axis=1) > 0.5)
    assert lit_rows.size >= 3
    assert bounds is not None
    assert bounds[1] <= lit_rows.min() + 0.5
    assert bounds[3] >= lit_rows.max() + 0.5
    assert bounds[3] - bounds[1] <= lit_rows.max() - lit_rows.min() + 2.0


@pytest.mark.scipy
def test_box_bounds_folded_readout_at_frame_edge() -> None:
    """A copy of the object cut off by the top of the frame pulls the box to that edge.

    The first 31 rows read source rows 20.3 to 23.3, inside the object, before the readout
    folds back, so the object is already showing at row 0.
    """
    h = 100
    rows = np.arange(h, dtype=np.float64)
    read = np.empty(h)
    read[:31] = 20.3 + 0.1 * rows[:31]
    read[31:61] = np.linspace(read[30], 5.0, 31)[1:]
    read[61:] = np.linspace(5.0, 95.0, 40)[1:]
    dy_rows = rows - read
    z = np.zeros(h)
    img = np.zeros((h, h))
    img[10:91, 50] = 1.0
    out = _render.warp_frame(image=img, dx_rows=z, dy_rows=dy_rows, roll_rad=0.0, ifov=1e-6, order=1, fill=np.zeros(1))
    coords = _render.source_coords(
        dx_rows=z,
        dy_rows=dy_rows,
        roll_rad=0.0,
        ifov=1e-6,
        height=h,
        width=h,
        border=True,
    )
    bounds = _render.box_bounds(coords=coords, boxes=[(48.0, 8.0, 53.0, 93.0)])[0]
    lit_rows = np.flatnonzero(out.max(axis=1) > 0.4)
    assert lit_rows.min() == 0
    assert bounds is not None
    assert bounds[1] == 0.0
    assert bounds[3] >= lit_rows.max() + 0.5


@pytest.mark.scipy
def test_keystone_matches_exact_projection() -> None:
    """The linearized keystone terms track the exact tan-projection at ~10 deg half-FOV."""
    ifov, dx = 2e-3, 5.0  # theta_yaw = 0.01 rad
    theta = dx * ifov
    u_c = 90.0
    exact_du = (np.tan(np.arctan(u_c * ifov) + theta) / ifov) - u_c
    z = np.zeros(281)
    dx_field, _dy_field = _render._displacement_fields(dx_rows=z + dx, dy_rows=z, ifov=ifov, height=281, width=281)
    assert dx_field[140, int(u_c + 140.0)] == pytest.approx(exact_du, abs=0.1)


@pytest.mark.scipy
def test_warp_fill_value_on_revealed_edge() -> None:
    """Edge pixels revealed by displacement take the fill value."""
    img = np.full((32, 32), 100.0)
    z = np.zeros(32)
    out = _render.warp_frame(
        image=img,
        dx_rows=z + 10.0,
        dy_rows=z,
        roll_rad=0.0,
        ifov=1e-6,
        order=1,
        fill=np.array([7.0]),
    )
    assert float(out[16, 0]) == pytest.approx(7.0)  # revealed left edge takes the fill value
