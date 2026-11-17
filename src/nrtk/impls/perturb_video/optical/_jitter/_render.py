"""Defines stateless rendering primitives for the FMV jitter perturber: angles to pixels.

Functions:
    directional_smear_psf: Rasterize a within-exposure LOS path into a motion-blur kernel.
    cast_back: Cast a float result back to the frame's dtype.
    convolve_psf: FFT-convolve an image with a PSF per channel.
    source_coords: Inverse map of the warp, the source coordinate each output pixel reads.
    warp_frame: Apply displacement + roll + keystone (+ per-row rolling shutter) as one warp.
    box_bounds: Output extent of source boxes under the same inverse map.

Dependencies:
    - numpy for array operations.
    - scipy for convolution and coordinate-map interpolation.

The warp is defined by its inverse map. Each output pixel reads the source coordinate found by
undoing the displacement field (per-row rolling-shutter displacement + linearized tip/tilt
keystone terms) at that pixel's own row, then undoing the roll about the image center
(positive ``roll_rad`` is counter-clockwise as displayed). ``source_coords`` builds the map,
``warp_frame`` resamples the image through it and ``box_bounds`` moves boxes with it, so the
image and its boxes share one geometry.
"""

from __future__ import annotations

__all__ = ["box_bounds", "cast_back", "convolve_psf", "directional_smear_psf", "source_coords", "warp_frame"]

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import ndimage, signal

_MIN_BOX_PX = 1e-9  # output boxes narrower than this are dropped


def directional_smear_psf(
    *,
    path_px_x: np.ndarray[Any, Any],
    path_px_y: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any] | None:
    """Rasterize a centered within-exposure pixel path into a normalized motion-blur PSF.

    After Boracchi & Foi (2012): the within-exposure LOS path, uniformly weighted in time, IS
    the motion-blur kernel. Bilinear splatting preserves the path's first moment; splats beyond
    the 99th-percentile kernel extent are clipped to the border (energy preserved; worst-case
    centroid motion ~0.05 px). Returns None when the excursion is sub-pixel (< 0.5 px).

    Args:
        path_px_x: Mean-subtracted within-exposure path, x (columns), in pixels.
        path_px_y: Mean-subtracted within-exposure path, y (rows), in pixels.

    Returns:
        (K, K) float64 kernel summing to 1, or None for sub-pixel excursions.

    Examples:
        >>> import numpy as np
        >>> k = directional_smear_psf(path_px_x=np.linspace(-2, 2, 64), path_px_y=np.zeros(64))
        >>> bool(abs(k.sum() - 1.0) < 1e-9)
        True
    """
    if path_px_x.size < 2:
        return None
    extent = float(np.percentile(np.hypot(path_px_x, path_px_y), 99))
    if extent < 0.5:
        return None
    half = int(np.ceil(extent)) + 1
    size = 2 * half + 1
    fx = np.clip(half + path_px_x, 0.0, size - 1.000001)
    fy = np.clip(half + path_px_y, 0.0, size - 1.000001)
    x0 = np.floor(fx).astype(np.intp)
    y0 = np.floor(fy).astype(np.intp)
    ax, ay = fx - x0, fy - y0
    kernel = np.zeros((size, size), dtype=np.float64)
    np.add.at(kernel, (y0, x0), (1.0 - ax) * (1.0 - ay))
    np.add.at(kernel, (y0, x0 + 1), ax * (1.0 - ay))
    np.add.at(kernel, (y0 + 1, x0), (1.0 - ax) * ay)
    np.add.at(kernel, (y0 + 1, x0 + 1), ax * ay)
    return kernel / kernel.sum()


def cast_back(*, out: np.ndarray[Any, Any], dtype: np.dtype) -> np.ndarray[Any, Any]:
    """Cast float results back to the original dtype: round + clip for integer dtypes.

    Out-of-range float->int casting is undefined behavior in numpy (wraps to bright/dark
    speckle); the cubic warp overshoots a 255-contrast edge to roughly [-30, +311], so the
    clip is mandatory, and rint avoids the truncation bias of a bare astype.
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(dtype)


def convolve_psf(*, image: np.ndarray[Any, Any], psf: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Convolve an (H, W) or (H, W, C) image with a 2-D PSF per channel, preserving shape/dtype.

    Uses true FFT *convolution* (``scipy.signal.fftconvolve``); note that correlation-based
    filters (e.g. cv2.filter2D) would mirror asymmetric kernels — this does not. The
    ``mode="same"`` zero boundary dims a kernel-half-width band at the frame edges, matching
    the convolution behavior of ``TurbulenceVideoPerturber``.

    Examples:
        >>> import numpy as np
        >>> img = np.zeros((8, 8), dtype=np.uint8)
        >>> out = convolve_psf(image=img, psf=np.ones((3, 3)) / 9.0)
        >>> bool(out.shape == (8, 8) and out.dtype == np.uint8)
        True
    """
    img = image.astype(np.float64)
    if img.ndim == 2:
        out = signal.fftconvolve(in1=img, in2=psf, mode="same")
    else:
        out = np.stack(
            [signal.fftconvolve(in1=img[..., c], in2=psf, mode="same") for c in range(img.shape[2])],
            axis=-1,
        )
    return cast_back(out=out, dtype=image.dtype)


def _sample_positions(*, size: int, border: bool) -> np.ndarray[Any, Any]:
    """Pixel-center indices along one axis, optionally with the two frame edges added."""
    centers = np.arange(size, dtype=np.float64)
    if not border:
        return centers
    return np.concatenate([[-0.5], centers, [size - 0.5]])


def _displacement_fields(
    *,
    dx_rows: np.ndarray[Any, Any],
    dy_rows: np.ndarray[Any, Any],
    ifov: float,
    height: int,
    width: int,
    border: bool = False,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Expand per-row displacements into per-pixel fields with linearized homography terms.

    Keystone (exact K.R.K^-1 to first order in the tilt angle):
    ``Dx = dx_row*(1 + ifov^2*u_c^2) + dy_row*ifov^2*u_c*v_c`` (and symmetrically for Dy),
    where u_c/v_c are center-relative pixel coordinates. Negligible at narrow FOV; ~2 px at
    the corners for wide-FOV handheld amplitudes — the same order as the rendered roll effect.
    With ``border`` the fields also cover the frame edges, see :func:`source_coords`.
    """
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    u_c = _sample_positions(size=width, border=border) - cx  # (W,)
    v_c = (_sample_positions(size=height, border=border) - cy)[:, None]  # (H, 1)
    if border:
        dx_rows = np.pad(dx_rows, 1, mode="edge")
        dy_rows = np.pad(dy_rows, 1, mode="edge")
    dxr = dx_rows[:, None]
    dyr = dy_rows[:, None]
    k2 = ifov * ifov
    dx_field = dxr * (1.0 + k2 * u_c**2) + dyr * k2 * u_c * v_c
    dy_field = dyr * (1.0 + k2 * v_c**2) + dxr * k2 * u_c * v_c
    return dx_field, dy_field


def source_coords(
    *,
    dx_rows: np.ndarray[Any, Any],
    dy_rows: np.ndarray[Any, Any],
    roll_rad: float,
    ifov: float,
    height: int,
    width: int,
    border: bool = False,
) -> np.ndarray[Any, Any]:
    """Source coordinate read by each output pixel (the inverse map of the warp).

    Each output pixel is moved back by the displacement field at its own row, then rotated
    back about the image center. Using the output row is what rolling shutter does: sensor
    row r reads the scene displaced at row r's exposure time. For the keystone terms it is a
    first-order inverse.

    Args:
        dx_rows: (H,) per-output-row x-displacement in pixels (constant array = global shutter).
        dy_rows: (H,) per-output-row y-displacement in pixels.
        roll_rad: Signed roll angle about the image center; positive rotates
            counter-clockwise as displayed, negative clockwise.
        ifov: Radians per pixel (drives the keystone terms).
        height: Image height (rows).
        width: Image width (columns).
        border: Also evaluate the map on the frame edges, half a pixel outside the first and
            last pixel centers, so that the samples span the whole frame. The edge rows use
            the displacement of the nearest image row.

    Returns:
        (2, H, W) float64 array of (source row, source column) per output pixel, the layout
        ``scipy.ndimage.map_coordinates`` takes. With ``border`` the shape is (2, H + 2, W + 2).

    Examples:
        >>> import numpy as np
        >>> z = np.zeros(3)
        >>> c = source_coords(dx_rows=z + 1.0, dy_rows=z, roll_rad=0.0, ifov=0.0, height=3, width=3)
        >>> [float(v) for v in c[1, 0]]  # each output column reads the source one column to its left
        [-1.0, 0.0, 1.0]
    """
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    dx_field, dy_field = _displacement_fields(
        dx_rows=dx_rows,
        dy_rows=dy_rows,
        ifov=ifov,
        height=height,
        width=width,
        border=border,
    )
    u1 = (_sample_positions(size=width, border=border)[None, :] - cx) - dx_field
    v1 = (_sample_positions(size=height, border=border)[:, None] - cy) - dy_field
    cos_a, sin_a = float(np.cos(roll_rad)), float(np.sin(roll_rad))
    return np.stack([sin_a * u1 + cos_a * v1 + cy, cos_a * u1 - sin_a * v1 + cx])


def warp_frame(
    *,
    image: np.ndarray[Any, Any],
    dx_rows: np.ndarray[Any, Any],
    dy_rows: np.ndarray[Any, Any],
    roll_rad: float,
    ifov: float,
    order: int,
    fill: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Resample a frame through the inverse map from :func:`source_coords`.

    Uses ``scipy.ndimage.map_coordinates``. ``order=3`` = cubic sub-pixel; ``order=0`` = nearest.

    Args:
        image: (H, W) or (H, W, C) array, any dtype (integer dtypes are rint+clip cast back).
        dx_rows: (H,) per-output-row x-displacement in pixels (constant array = global shutter).
        dy_rows: (H,) per-output-row y-displacement in pixels.
        roll_rad: Signed roll angle about the image center; positive rotates
            counter-clockwise as displayed, negative clockwise.
        ifov: Radians per pixel (drives the keystone terms).
        order: Spline interpolation order for ``map_coordinates``.
        fill: Per-channel fill values for revealed edge pixels, shape (C,) or (1,) for 2-D.

    Returns:
        Warped array, same shape and dtype as ``image``.

    Examples:
        >>> import numpy as np
        >>> img = np.zeros((5, 5))
        >>> img[2, 3] = 1.0
        >>> z = np.zeros(5)
        >>> out = warp_frame(image=img, dx_rows=z + 1.0, dy_rows=z, roll_rad=0.0, ifov=1e-6, order=1, fill=np.zeros(1))
        >>> int(np.argmax(out[2]))
        4
    """
    h, w = image.shape[:2]
    coords = source_coords(dx_rows=dx_rows, dy_rows=dy_rows, roll_rad=roll_rad, ifov=ifov, height=h, width=w)

    def _warp_plane(*, plane: np.ndarray[Any, Any], cval: float) -> np.ndarray[Any, Any]:
        return ndimage.map_coordinates(
            input=plane.astype(np.float64),
            coordinates=coords,
            order=order,
            mode="constant",
            cval=cval,
            prefilter=True,
        )

    if image.ndim == 2:
        out = _warp_plane(plane=image, cval=float(fill[0]))
    else:
        out = np.stack(
            [_warp_plane(plane=image[..., c], cval=float(fill[min(c, fill.size - 1)])) for c in range(image.shape[2])],
            axis=-1,
        )
    return cast_back(out=out, dtype=image.dtype)


def _clip_segments(
    *,
    start: np.ndarray[Any, Any],
    stop: np.ndarray[Any, Any],
    lo: np.ndarray[Any, Any],
    hi: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Clip the segments ``start -> stop`` to the rectangle ``[lo, hi]`` (Liang-Barsky).

    Args:
        start: (2, N) segment start points.
        stop: (2, N) segment end points.
        lo: (2,) rectangle minimum.
        hi: (2,) rectangle maximum.

    Returns:
        ``(t0, t1)``, the part of each segment inside the rectangle as fractions of its length.
        A segment that misses the rectangle has ``t0 > t1``.
    """
    lo, hi = lo[:, None], hi[:, None]
    step = stop - start
    with np.errstate(divide="ignore", invalid="ignore"):
        ta = (lo - start) / step
        tb = (hi - start) / step
    flat = step == 0
    within = (start >= lo) & (start <= hi)
    t_in = np.where(flat, np.where(within, -np.inf, np.inf), np.minimum(ta, tb))
    t_out = np.where(flat, np.where(within, np.inf, -np.inf), np.maximum(ta, tb))
    return np.maximum(t_in.max(axis=0), 0.0), np.minimum(t_out.min(axis=0), 1.0)


def _clipped_edge_ends(
    *,
    grid: np.ndarray[Any, Any],
    outside: np.ndarray[Any, Any],
    lo: np.ndarray[Any, Any],
    hi: np.ndarray[Any, Any],
    step: tuple[int, int],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Clip the grid edges that run in direction ``step`` to the rectangle ``[lo, hi]``.

    Args:
        grid: (2, R, C) source coordinates.
        outside: (R, C) outcodes of ``grid`` against the rectangle (0 = inside).
        lo: (2,) rectangle minimum.
        hi: (2,) rectangle maximum.
        step: ``(row, column)`` offset from one end of an edge to the other.

    Returns:
        Fractional ``(rows, columns)`` grid indices of both ends of every clipped edge.
    """
    d_row, d_col = step
    first = outside[: outside.shape[0] - d_row, : outside.shape[1] - d_col]
    second = outside[d_row:, d_col:]
    # Edges with both ends inside add nothing new, and ends beyond the same side cannot cross.
    r, c = np.nonzero(((first | second) != 0) & ((first & second) == 0))
    t0, t1 = _clip_segments(start=grid[:, r, c], stop=grid[:, r + d_row, c + d_col], lo=lo, hi=hi)
    hit = t0 <= t1
    t = np.concatenate([t0[hit], t1[hit]])
    return np.tile(r[hit], 2) + d_row * t, np.tile(c[hit], 2) + d_col * t


def _cross(*, a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """2-D cross product of (2, N) vectors."""
    return a[0] * b[1] - a[1] * b[0]


def _corner_positions(
    *,
    grid: np.ndarray[Any, Any],
    outside: np.ndarray[Any, Any],
    lo: np.ndarray[Any, Any],
    hi: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Locate the corners of the rectangle ``[lo, hi]`` on the grid.

    Each grid cell is split along its diagonal into two triangles on which the map is linear.

    Args:
        grid: (2, R, C) source coordinates.
        outside: (R, C) outcodes of ``grid`` against the rectangle (0 = inside).
        lo: (2,) rectangle minimum.
        hi: (2,) rectangle maximum.

    Returns:
        Fractional ``(rows, columns)`` grid indices that map onto a corner.
    """
    codes = (outside[:-1, :-1], outside[:-1, 1:], outside[1:, :-1], outside[1:, 1:])
    # sides of the rectangle that pass through each cell
    split = (codes[0] | codes[1] | codes[2] | codes[3]) & ~(codes[0] & codes[1] & codes[2] & codes[3])
    rows, cols = [], []
    for row_bit, y in ((1, lo[0]), (2, hi[0])):
        for col_bit, x in ((4, lo[1]), (8, hi[1])):
            r, c = np.nonzero((split & (row_bit | col_bit)) == (row_bit | col_bit))
            p00, p01, p10, p11 = grid[:, r, c], grid[:, r, c + 1], grid[:, r + 1, c], grid[:, r + 1, c + 1]
            d = np.array([[y], [x]]) - p00
            # u runs along columns and v along rows; v <= u in the upper triangle, u <= v in the lower
            for e_u, e_v, upper in ((p01 - p00, p11 - p01, True), (p11 - p10, p10 - p00, False)):
                with np.errstate(divide="ignore", invalid="ignore"):
                    det = _cross(a=e_u, b=e_v)
                    u = _cross(a=d, b=e_v) / det
                    v = _cross(a=e_u, b=d) / det
                hit = (v >= 0) & (v <= u) & (u <= 1) if upper else (u >= 0) & (u <= v) & (v <= 1)
                rows.append(r[hit] + v[hit])
                cols.append(c[hit] + u[hit])
    return np.concatenate(rows), np.concatenate(cols)


def _pair_span(*, lo: np.ndarray[Any, Any], hi: np.ndarray[Any, Any], a: float, b: float) -> tuple[int, int] | None:
    """Slice bounds covering every adjacent pair of entries whose joint range ``[lo, hi]`` meets ``[a, b]``."""
    hit = np.flatnonzero((np.minimum(lo[:-1], lo[1:]) <= b) & (np.maximum(hi[:-1], hi[1:]) >= a))
    if hit.size == 0:
        return None
    return int(hit[0]), int(hit[-1]) + 2


def box_bounds(
    *,
    coords: np.ndarray[Any, Any],
    boxes: Sequence[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float] | None]:
    """Output extent of source boxes under the warp described by ``coords``.

    Box coordinates are pixel edges, so pixel ``i`` spans ``[i, i + 1]`` and a full-frame box
    is ``(0, 0, W, H)``. The inverse map is interpolated linearly between its samples, on
    triangles, and the result is the bounding box of everything in the frame that maps into
    the source box. That region is a set of polygons whose vertices are the samples inside the
    box, the points where grid edges cross the box boundary, and the box corners. A folded
    rolling-shutter readout shows a box more than once and every copy is included.

    Args:
        coords: ``source_coords(..., border=True)`` output, (2, H + 2, W + 2).
        boxes: Source boxes as ``(x1, y1, x2, y2)``.

    Returns:
        One ``(x1, y1, x2, y2)`` per box, or None where the box does not appear in the frame.

    Examples:
        >>> import numpy as np
        >>> z = np.zeros(5)
        >>> c = source_coords(dx_rows=z + 1.0, dy_rows=z, roll_rad=0.0, ifov=0.0, height=5, width=5, border=True)
        >>> box_bounds(coords=c, boxes=[(1.0, 2.0, 2.0, 3.0)])
        [(2.0, 2.0, 3.0, 3.0)]
    """
    # pixel-edge coordinate of each grid row and column
    row_pos = _sample_positions(size=coords.shape[1] - 2, border=True) + 0.5
    col_pos = _sample_positions(size=coords.shape[2] - 2, border=True) + 0.5
    row_lo, row_hi = coords[0].min(axis=1), coords[0].max(axis=1)
    col_lo, col_hi = coords[1].min(axis=0), coords[1].max(axis=0)

    out: list[tuple[float, float, float, float] | None] = []
    for x1, y1, x2, y2 in boxes:
        # coords index pixel centers, half a pixel in from the edges the box is given in
        lo = np.array([y1, x1]) - 0.5
        hi = np.array([y2, x2]) - 0.5
        rows = _pair_span(lo=row_lo, hi=row_hi, a=lo[0], b=hi[0])
        cols = _pair_span(lo=col_lo, hi=col_hi, a=lo[1], b=hi[1])
        if rows is None or cols is None:
            out.append(None)
            continue
        grid = coords[:, rows[0] : rows[1], cols[0] : cols[1]]
        # one bit per box side a sample lies beyond
        outside = np.zeros(grid.shape[1:], dtype=np.uint8)
        for bit, beyond in enumerate((grid[0] < lo[0], grid[0] > hi[0], grid[1] < lo[1], grid[1] > hi[1])):
            outside |= beyond.astype(np.uint8) << bit
        inside = outside == 0
        found = [(np.flatnonzero(inside.any(axis=1)), np.flatnonzero(inside.any(axis=0)))]
        found += [
            _clipped_edge_ends(grid=grid, outside=outside, lo=lo, hi=hi, step=step) for step in ((0, 1), (1, 0), (1, 1))
        ]
        found.append(_corner_positions(grid=grid, outside=outside, lo=lo, hi=hi))
        r = np.concatenate([f[0] for f in found]) + rows[0]
        c = np.concatenate([f[1] for f in found]) + cols[0]
        if r.size == 0 or c.size == 0:
            out.append(None)
            continue
        ny1, ny2 = (float(v) for v in np.interp([r.min(), r.max()], np.arange(row_pos.size), row_pos))
        nx1, nx2 = (float(v) for v in np.interp([c.min(), c.max()], np.arange(col_pos.size), col_pos))
        # a box that only touches the frame comes out with rounding-sized width or height
        out.append((nx1, ny1, nx2, ny2) if nx2 - nx1 > _MIN_BOX_PX and ny2 - ny1 > _MIN_BOX_PX else None)
    return out
