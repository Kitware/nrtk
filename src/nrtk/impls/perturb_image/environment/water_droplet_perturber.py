# This file includes code originally posted on Stack Overflow
# (https://stackoverflow.com/posts/50751932/revisions) and is licensed under the
# Creative Commons Attribution-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-sa/4.0/
#
# © 2025 Kitware, Inc

"""Defines WaterDropletPerturber for physics-based, photorealistic water droplet effects on images.

WaterDropletPerturber Implementation based on the work from these source papers:
(1)
https://openaccess.thecvf.com/content_ICCVW_2019/papers/PBDL/Hao_Learning_From_Synthetic_Photorealistic_Raindrop_for_Single_Image_Raindrop_Removal_ICCVW_2019_paper.pdf
(2) https://www.giss.nasa.gov/pubs/abs/er05000f.html

For additional research regarding Water Droplet modeling, please refer to this paper:
https://www.cvlibs.net/publications/Roser2010ACCVWORK.pdf

Classes:
    WaterDropletPerturber:
        Implements the physics-based, photorealistic water/rain droplet model, utilizing
        Scipy and Numba functionalities.
    Bezier:
        Implements a class to compute Bezier curve based on the segment information

Dependencies:
    - Scipy for image processing.
    - Numba for JIT-compiled point-in-polygon operations.
    - nrtk.interfaces.perturb_image.PerturbImage as the base interface for image perturbation.

Example usage:
    >>> if not WaterDropletPerturber.is_usable():
    ...     import pytest
    ...
    ...     pytest.skip("WaterDropletPerturber is not usable")
    >>> size_range = (0.0, 1.0)
    >>> num_drops = 20
    >>> perturber = WaterDropletPerturber(size_range=size_range, num_drops=num_drops)
    >>> image = np.ones((256, 256, 3))
    >>> perturbed_image, _ = perturber(image=image)

Notes:
    - The boxes returned from `perturb` are identical to the boxes passed in.
"""

from __future__ import annotations

__all__ = ["WaterDropletPerturber"]

import copy
import math
from collections.abc import Hashable, Iterable, Sequence
from typing import Any, Protocol

from typing_extensions import override

from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.utils._exceptions import WaterDropletImportError
from nrtk.utils._import_guard import import_guard

scipy_available: bool = import_guard(
    module_name="scipy",
    exception=WaterDropletImportError,
    submodules=["special", "spatial", "ndimage"],
)

# NOTE: Using try-except instead of import_guard for numba to avoid a conflict
# with smqtk_detection's centernet.py. The import_guard creates a mock module in
# sys.modules, which causes centernet.py's `find_spec('numba')` call to fail with
# "ValueError: numba.__spec__ is not set". This will be resolved when we switch
# to the new import guard approach.
numba_available: bool
try:
    import numba

    numba_available = True
except ImportError:
    numba = None  # type: ignore[assignment]
    numba_available = False

import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.spatial import KDTree  # noqa: E402
from scipy.special import binom  # noqa: E402
from smqtk_image_io.bbox import AxisAlignedBoundingBox  # noqa: E402


def _points_in_polygon_impl(*, points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Check if points are inside a polygon using ray casting algorithm.

    This is the pure Python implementation of the ray casting algorithm
    for point-in-polygon tests. For each point, it casts a horizontal ray
    to the right and counts edge crossings. Odd count = inside.

    Args:
        points: Array of shape (N, 2) containing x, y coordinates to test.
        polygon: Array of shape (M, 2) containing polygon vertices in order.

    Returns:
        Boolean array of shape (N,) indicating which points are inside.
    """
    n_points = points.shape[0]
    n_verts = polygon.shape[0]
    result = np.zeros(n_points, dtype=np.bool_)

    for i in range(n_points):
        px, py = points[i, 0], points[i, 1]
        inside = False

        j = n_verts - 1
        for k in range(n_verts):
            vkx, vky = polygon[k, 0], polygon[k, 1]
            vjx, vjy = polygon[j, 0], polygon[j, 1]

            # Check if ray from point crosses this edge
            if ((vky > py) != (vjy > py)) and (px < (vjx - vkx) * (py - vky) / (vjy - vky) + vkx):
                inside = not inside

            j = k

        result[i] = inside

    return result


def _compute_refraction_mapping_impl(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    gls: np.ndarray,
    normal: np.ndarray,
    n_air: float,
    n_water: float,
    M: int,  # noqa: N803
    B: int,  # noqa: N803
    center: np.ndarray,
    radius: float,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute refraction mapping for all pixels in a droplet.

    This is the pure Python implementation that computes where each pixel maps to
    after refraction through the water droplet.

    Args:
        xs: Array of x coordinates for pixels in the droplet.
        ys: Array of y coordinates for pixels in the droplet.
        gls: Glass coordinate system mapping matrix.
        normal: Normal vector to the glass plane.
        n_air: Refractive index of air.
        n_water: Refractive index of water.
        M: Distance to glass plane in cm.
        B: Distance to background plane in cm.
        center: Center of the droplet sphere.
        radius: Radius of the droplet sphere.
        intrinsic: Camera intrinsic matrix.

    Returns:
        Tuple of (u_coords, v_coords) arrays with mapped pixel coordinates.
    """
    n_pixels = xs.shape[0]
    u_out = np.empty(n_pixels, dtype=np.float64)
    v_out = np.empty(n_pixels, dtype=np.float64)

    # Precompute constants
    normal_dot = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]
    o_g = (normal[2] * M / normal_dot) * normal
    gamma = np.arcsin(n_air / n_water)

    for i in range(n_pixels):
        x = xs[i]
        y = ys[i]
        p_g = gls[x, y]

        # Compute alpha and beta for refraction at glass surface
        p_g_norm = np.sqrt(p_g[0] ** 2 + p_g[1] ** 2 + p_g[2] ** 2)
        p_g_dot_normal = p_g[0] * normal[0] + p_g[1] * normal[1] + p_g[2] * normal[2]
        alpha = np.arccos(p_g_dot_normal / p_g_norm)
        beta = np.arcsin(n_air * np.sin(alpha) / n_water)

        # Compute refracted ray direction at glass surface
        po = p_g - o_g
        po_norm = np.sqrt(po[0] ** 2 + po[1] ** 2 + po[2] ** 2)
        po = po / po_norm
        tan_beta = np.tan(beta)
        i_1 = normal + tan_beta * po
        i_1_norm = np.sqrt(i_1[0] ** 2 + i_1[1] ** 2 + i_1[2] ** 2)
        i_1 = i_1 / i_1_norm

        # Find intersection with sphere
        oc = p_g - center
        tmp = i_1[0] * oc[0] + i_1[1] * oc[1] + i_1[2] * oc[2]
        oc_dot = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2]
        d = -tmp + np.sqrt(abs(tmp**2 - oc_dot + radius**2))
        p_w = p_g + d * i_1

        # Compute normal at sphere surface
        normal_w = p_w - center
        normal_w_norm = np.sqrt(normal_w[0] ** 2 + normal_w[1] ** 2 + normal_w[2] ** 2)
        normal_w = normal_w / normal_w_norm

        # Compute tangent direction
        nw_dot_nw = normal_w[0] * normal_w[0] + normal_w[1] * normal_w[1] + normal_w[2] * normal_w[2]
        pw_dot_nw = p_w[0] * normal_w[0] + p_w[1] * normal_w[1] + p_w[2] * normal_w[2]
        pg_dot_nw = p_g[0] * normal_w[0] + p_g[1] * normal_w[1] + p_g[2] * normal_w[2]
        d2 = (pw_dot_nw - pg_dot_nw) / nw_dot_nw
        p_a = p_w - (d2 * normal_w + p_g)
        p_a_norm = np.sqrt(p_a[0] ** 2 + p_a[1] ** 2 + p_a[2] ** 2)
        if p_a_norm > 1e-10:
            p_a = p_a / p_a_norm

        # Compute angle of incidence at sphere
        pw_minus_pg = p_w - p_g
        pw_pg_norm = np.sqrt(pw_minus_pg[0] ** 2 + pw_minus_pg[1] ** 2 + pw_minus_pg[2] ** 2)
        nw_dot_pwpg = normal_w[0] * pw_minus_pg[0] + normal_w[1] * pw_minus_pg[1] + normal_w[2] * pw_minus_pg[2]
        eta = np.arccos(nw_dot_pwpg / pw_pg_norm)

        # Clamp eta to avoid total internal reflection
        if eta >= gamma:
            eta = gamma - 0.2

        # Compute refracted ray at sphere surface
        theta = np.arcsin(n_water * np.sin(eta) / n_air)
        i_2 = normal_w + np.tan(theta) * p_a
        p_e = p_w + ((B - p_w[2]) / i_2[2]) * i_2

        # Project to image plane
        u_out[i] = np.round((intrinsic[0, 0] * p_e[0] + intrinsic[0, 2] * p_e[2]) / B)
        v_out[i] = np.round((intrinsic[1, 1] * p_e[1] + intrinsic[1, 2] * p_e[2]) / B)

    return u_out, v_out


# Protocol types for the JIT-compiled or pure Python functions
class _PointsInPolygonProtocol(Protocol):
    """Protocol for point-in-polygon function signature."""

    def __call__(self, *, points: np.ndarray, polygon: np.ndarray) -> np.ndarray: ...


class _ComputeRefractionMappingProtocol(Protocol):
    """Protocol for refraction mapping function signature."""

    def __call__(
        self,
        *,
        xs: np.ndarray,
        ys: np.ndarray,
        gls: np.ndarray,
        normal: np.ndarray,
        n_air: float,
        n_water: float,
        M: int,  # noqa: N803
        B: int,  # noqa: N803
        center: np.ndarray,
        radius: float,
        intrinsic: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]: ...


# Apply numba JIT compilation when available for performance
_points_in_polygon: _PointsInPolygonProtocol
_compute_refraction_mapping: _ComputeRefractionMappingProtocol

if numba_available:  # pragma: no cover - JIT compilation not tracked by coverage
    _points_in_polygon = numba.njit(cache=True)(_points_in_polygon_impl)  # type: ignore[union-attr]
    _compute_refraction_mapping = numba.njit(  # type: ignore[union-attr]
        cache=True,
        parallel=True,
    )(_compute_refraction_mapping_impl)
else:  # pragma: no cover - fallback when numba not installed
    _points_in_polygon = _points_in_polygon_impl
    _compute_refraction_mapping = _compute_refraction_mapping_impl


class Bezier:
    """Class that computes the Bezier curve based on the segment information.

    Each curve is made of a series of segments that are initialized by the
    input points, angles, target radius and number of points needed for the
    Bezier interpolation.
    """

    def __init__(
        self,
        *,
        p1: np.ndarray[Any, Any],
        p2: np.ndarray[Any, Any],
        angle1: float,
        angle2: float,
        r: float = 0.3,
        num_points: int = 100,
    ) -> None:
        """Define segment parameters - points, angles, radius.

        Attributes:
            p1 (np.array): Start point of Bezier curve.
            p2 (np.array): End point of Bezier curve.
            angle1 (float): Direction angle (in radians) of tangent angle from p1
            angle2 (float): Direction angle (in radians) of tangent angle from p2
            r (float): scaling factor of curve
            num_points (int): number of points in curve
        """
        self.p1 = p1
        self.p2 = p2
        self.angle1 = angle1
        self.angle2 = angle2
        self.r = r
        self.num_points = num_points
        self._calc_intermediate_points()

    def _calc_intermediate_points(self) -> None:
        d = np.sqrt(np.sum((self.p2 - self.p1) ** 2))
        p = np.zeros((4, 2))
        p[0, :] = self.p1[:]
        p[3, :] = self.p2[:]
        p[1, :] = self.p1 + np.array([self.r * d * np.cos(self.angle1), self.r * d * np.sin(self.angle1)])
        p[2, :] = self.p2 + np.array(
            [self.r * d * np.cos(self.angle2 + np.pi), self.r * d * np.sin(self.angle2 + np.pi)],
        )
        self.p = p

    def get_curve(self) -> np.ndarray[Any, Any]:
        """Returns curve information."""
        return self.bezier()

    def bezier(self) -> np.ndarray[Any, Any]:
        """Draw Bézier curve by interpolating segments based on the Bernstein basis poynomial function.

        The Bezier curve equation is derived by combining the Bernstein basis polynomials with the control points:
            B(t) =  ∑(i=0 to n) B(i, n)(t) * P(i)
        """

        def _bernstein(*, n: int, k: int, t: np.ndarray) -> np.ndarray:
            """Bernstein basis polynomial function.

            Defined as: B(k, n)(t) = (n! / (k! * (n-k)!)) * t^k * (1 - t)^(n - k)

            n is the degree of the curve
            k is the index of the control point
            t is a parameter that varies from 0 to 1, defining the position along the curve.

            For n=2, the Bernstein polynomials are:
            - B(0, 2)(t) = (1 - t)^2
            - B(1, 2)(t) = 2t(1 - t)
            - B(2, 2)(t) = t^2

            For n=3, the Bernstein polynomials are:
            - B(0, 3)(t) = (1 - t)^3
            - B(1, 3)(t) = 3t(1 - t)^2
            - B(2, 3)(t) = 3t^2(1 - t)
            - B(3, 3)(t) = t^3.

            """
            return binom(n, k) * t**k * (1.0 - t) ** (n - k)

        n = len(self.p)
        t = np.linspace(0, 1, num=self.num_points)
        curve = np.zeros((self.num_points, 2))
        for i in range(n):
            curve += np.outer(
                _bernstein(n=n - 1, k=i, t=t),
                self.p[i],
            )
        return curve


class WaterDropletPerturber(PerturbImage):
    """Implements the physics-based, photorealistic water/rain droplet model.

    The `WaterDropletPerturber` class simulates the effects of rain/water droplets
    on an image similar to the rain drops on a window, car-windshield, etc. The equations
    defined for this model are based on the dynamics, geometry and photometry of a water/rain droplet.

    Attributes:
        size_range (Sequence[float]):
            Range of size multiplier values used for computing the size of the water droplet.
        num_drops (int):
            Target number of water droplets.
        blur_strength (float):
            Strength of Gaussian blur operation.
        psi (float):
            Angle between the camera line-of-sight and glass plane (radians).
        n_air (float):
            Density of air.
        n_water (float):
            Density of water.
        f_x (int):
            Camera focal length in x direction (cm).
        f_y (int):
            Camera focal length in y direction (cm).
        seed (int | None):
            Random seed for reproducibility. Defaults to 1.
    """

    def __init__(
        self,
        *,
        size_range: Sequence[float] = [0.0, 1.0],
        num_drops: int = 20,
        blur_strength: float = 0.25,
        psi: float = 90.0 / 180.0 * np.pi,
        n_air: float = 1.0,
        n_water: float = 1.33,
        f_x: int = 400,
        f_y: int = 400,
        seed: int | None = 1,
    ) -> None:
        """Initializes the WaterDropletPerturber.

        Args:
            size_range:
                Range of size multiplier values used for computing the size of the water droplet.
            num_drops:
                Target number of water droplets.
            blur_strength:
                Strength of Gaussian blur operation.
            psi:
                Angle between the line-of-sight and glass plane (radians).
            n_air:
                Density of air.
            n_water:
                Density of water.
            f_x:
                Camera focal length in x direction (cm).
            f_y:
                Camera focal length in y direction (cm).
            seed:
                Random seed for reproducible results. Defaults to 1 for deterministic behavior.

            If any of the parameters are absent, the following values will be set
            as defaults:

            size_range = (0.0, 1.0)
            num_drops = 20
            blur_strength = 0.25
            psi = 90.0 / 180.0 * np.pi
            n_air = 1.0
            n_water = 1.33
            f_x = 400
            f_y = 400
            seed = 1

        Raises:
            ImportError: If Scipy or Numba is not found.
        """
        if not self.is_usable():
            raise WaterDropletImportError
        super().__init__()
        self.size_range = size_range
        self.num_drops = num_drops
        self.blur_strength = blur_strength
        self.psi = psi
        self.n_air = n_air
        self.n_water = n_water
        self.f_x = f_x
        self.f_y = f_y
        self.seed = seed
        self._initialize_derived_parameters()

    def _initialize_derived_parameters(self) -> None:
        """Derived Parameters."""
        self.rng: np.random.Generator = np.random.default_rng(self.seed)
        # Glass plane at M centimeters ahead of the camera (value range chosen from source paper)
        self.M = self.rng.integers(low=20, high=40)

        # Background plane which is B centimeters from the camera
        # and lies beyond the glass plane (value range chosen from source paper)
        self.B = self.rng.integers(low=800, high=1500)

        self.normal: np.ndarray[Any, Any] = np.array([0.0, -1.0 * np.cos(self.psi), np.sin(self.psi)])

        self.g_centers = list()
        self.g_radius = list()
        self.centers = list()
        self.radius = list()

    @staticmethod
    def ccw_sort(points: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Returns sorted points in counterclockwise order around a center point."""
        # Subtract original point from center point (position obtained
        # by calculating the mean)
        points128 = points.astype(np.longdouble)
        d = points128 - np.mean(points128, axis=0)
        # Use atan2 to determine the angle taking into account the correct quadrant
        s = np.arctan2(d[:, 0], d[:, 1])
        # Return the sorted array of points.
        return points[np.argsort(s), :]

    @staticmethod
    def get_bezier_curve(
        *,
        points: np.ndarray[Any, Any],
        rad: float = 0.2,
        edgy: float = 0.0,
        tol: float = 1e-8,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Given an array of *points*, create a curve through those points.

        Args:
            points:
                arrays of points to create curve through
            rad:
                a number between 0 and 1 to steer the distance of control points.
            edgy:
                controls how "edgy" the curve is, edgy=0 is smoothest.
            tol:
                controls the tolerance used when comparing angles. Default is 1e-8.

        Returns:
            a tuple of point arrays that represent a Bezier curve
        """
        p = np.arctan(edgy) / np.pi + 0.5
        points = WaterDropletPerturber.ccw_sort(points)
        points = np.append(points, np.atleast_2d(points[0, :]), axis=0)
        d = np.diff(points, axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0])

        def _threshold_angle(ang: np.ndarray) -> np.ndarray:
            return (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)

        ang = _threshold_angle(ang)
        ang1 = ang
        ang2 = np.roll(ang, 1)
        ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > (np.pi + tol)) * np.pi
        ang = np.append(ang, [ang[0]])
        points = np.append(points, np.atleast_2d(ang).T, axis=1)

        def _get_curve(*, points: np.ndarray, r: float) -> np.ndarray:
            """Get the segments and curve data."""
            segments = list()
            for i in range(len(points) - 1):
                seg = Bezier(
                    p1=points[i, :2],
                    p2=points[i + 1, :2],
                    angle1=points[i, 2],
                    angle2=points[i + 1, 2],
                    r=r,
                )
                segments.append(seg.get_curve())

            return np.concatenate(segments)

        c = _get_curve(points=points, r=rad)
        x, y = c.T
        return x, y

    @staticmethod
    def get_random_points_within_min_dist(
        *,
        rng: np.random.Generator,
        n: int = 5,
        scale: float = 0.8,
        min_dst: float | None = None,
        recursive: int = 0,
    ) -> np.ndarray[Any, Any]:
        """Function to create *n* random points in the unit square, which are *min_dst* apart, then scale them.

        Args:
            rng:
                numpy random generator to use
            n:
                number of random points to create
            scale:
                how much to scale points once recursion is finished
            min_dst:
                minimum distance between the random points
            recursive:
                current number of recursive loops

        Returns:
            a random array of points within a minimum distance
        """
        min_dst = min_dst or 0.7 / n
        points = rng.random((n, 2))
        d = np.sqrt(np.sum(np.diff(WaterDropletPerturber.ccw_sort(points), axis=0), axis=1) ** 2)
        if np.all(d >= min_dst) or recursive >= 200:
            return points * scale
        return WaterDropletPerturber.get_random_points_within_min_dist(
            rng=rng,
            n=n,
            scale=scale,
            min_dst=min_dst,
            recursive=recursive + 1,
        )

    def _to_glass(
        self,
        *,
        x: int,
        y: int,
        psi: float,
        M: int,  # noqa: N803
        intrinsic: np.ndarray,
    ) -> np.ndarray:
        """Convert 2D pixel coordinates from an image (x, y) into a 3D point in the glass coordinate system.

        Args:
            x:
                X-coordinate value in the image plane.
            y:
                Y-coordinate value in the image plane.
            psi:
                Angle between the camera line-of-sight and glass plane (radians).
            M:
                Glass plane at M centimeters ahead of the camera.
            intrinsic:
                Intrinsic (Camera) parameters matrix.

        Returns:
            Glass plane (3D) coordinate system matrix.
        """
        xx, yy = np.meshgrid(np.arange(x), np.arange(y), indexing="ij")
        w = M * np.tan(psi) / (np.tan(psi) - (yy - intrinsic[1, 2]) / intrinsic[1, 1])
        u = w * (xx - intrinsic[0, 2]) / intrinsic[0, 0]
        v = w * (yy - intrinsic[1, 2]) / intrinsic[1, 1]
        return np.dstack((u, v, w)).reshape((x, y, 3))

    def _get_sphere_raindrop(self, *, width: int, height: int, gls: np.ndarray[Any, Any]) -> None:  # noqa: C901
        """Simulate and store information about raindrops on the windshield.

        How it works:
        - Defines random parameters for new raindrops, like size and position.
        - Generates raindrop spheres and their attributes (e.g., center, radius).
        - Stores this data for later use in the simulation.

        Args:
            width:
                Input image width.
            height:
                Input image height.
            gls:
                Glass (3D) coordinate system mapping matrix.
        """
        self.g_centers: list[Any] = list()
        self.g_radius: list[Any] = list()
        self.centers: list[Any] = list()
        self.radius: list[Any] = list()

        left_upper = gls[0][0]
        left_bottom = gls[0][height - 1]
        right_bottom = gls[width - 1][height - 1]

        def _random_tau() -> int:
            """Determines angle between tangent and glass plane."""
            return math.floor(self.rng.uniform(low=30, high=45))

        def _random_loc() -> float:
            """Determine random multiplier value that is applied to the water droplet size computation."""
            return self.rng.uniform(low=self.size_range[0], high=self.size_range[1])

        def _w_in_plane(*, u: int, v: int) -> int:
            """Estimate the "depth" value of a pixel in the coordinate system of the glass plane."""
            return (self.normal[2] * self.M - self.normal[0] * u - self.normal[1] * v) / self.normal[2]

        def _check_overlap(
            *,
            i: int,
            nearby: list[int],
            centers: np.ndarray,
            radii: np.ndarray,
            indices_to_remove: set[int],
        ) -> bool:
            """Check if droplet i overlaps with any nearby droplet."""
            for j in nearby:
                if i >= j or j in indices_to_remove:
                    continue
                if radii[i] + radii[j] > np.linalg.norm(centers[i] - centers[j]):
                    return True
            return False

        def _find_overlapping_indices(*, centers: np.ndarray, radii: np.ndarray) -> set[int]:
            """Find indices of droplets that overlap with others using KDTree spatial indexing."""
            tree = KDTree(data=centers)
            max_radius = radii.max()
            indices_to_remove: set[int] = set()

            for i in range(len(centers)):
                if i in indices_to_remove:
                    continue
                nearby = tree.query_ball_point(x=centers[i], r=radii[i] + max_radius)
                if _check_overlap(
                    i=i,
                    nearby=nearby,
                    centers=centers,
                    radii=radii,
                    indices_to_remove=indices_to_remove,
                ):
                    indices_to_remove.add(i)
            return indices_to_remove

        def _remove_overlapping_drops() -> None:
            """Remove overlapping droplet spheres using spatial indexing (KDTree).

            Uses O(n log n) spatial indexing instead of O(n²) pairwise comparison
            to efficiently detect and remove overlapping droplets.
            """
            if len(self.g_centers) < 2:
                return

            indices_to_remove = _find_overlapping_indices(
                centers=np.array(self.g_centers),
                radii=np.array(self.g_radius),
            )

            for index in sorted(indices_to_remove, reverse=True):
                self.g_centers.pop(index)
                self.g_radius.pop(index)
                self.centers.pop(index)
                self.radius.pop(index)

        for _ in range(self.num_drops):
            u = left_bottom[0] + (right_bottom[0] - left_bottom[0]) * self.rng.random()
            v = left_upper[1] + (right_bottom[1] - left_upper[1]) * self.rng.random()
            w = _w_in_plane(u=u, v=v)

            # Convert the angle between tangent and glass plane from degrees to radians
            tau = _random_tau() / 180 * np.pi

            # Water droplet size computation
            glass_r = 0.1 + (width // 500) * _random_loc()

            # raindrop radius in sky dataset
            r_sphere = glass_r / np.sin(tau)

            # g_c is the center of the raindrop in the windshield plane's coordinate system
            g_c = np.array([u, v, w])
            # c is the adjusted center (based on depth of drop itself)
            c = g_c - np.array([0, 0, r_sphere * np.cos(tau)])

            self.g_centers.append(g_c)
            self.g_radius.append(glass_r)
            self.centers.append(c)
            self.radius.append(r_sphere)
        _remove_overlapping_drops()

    def _in_sphere_raindrop(self, gls: np.ndarray) -> np.ndarray:
        """Helper function for rendering.

        Determine if a given pixel (x, y) is inside any simulated raindrop on
        the windshield. If yes, == index of the raindrop, if no, == -1.

        Args:
            gls: Glass (3D) coordinate system mapping matrix.

        Returns:
            Truth mask of valid pixels.
        """
        p = gls
        q = np.ones(p.shape[:2]) * -1
        for i, (center, radius) in enumerate(zip(self.g_centers, self.g_radius, strict=False)):
            dist = np.linalg.norm(p - center, axis=-1)
            # Give the true/false values of where the points of the image are within sphere
            # These values are in the coordinate system of the original image
            indices = dist <= radius

            q[indices] = i

            # Find the center of the droplet in terms of the image pixels
            min_idx = np.unravel_index(np.argmin(dist), dist.shape)
            x_cent, y_cent = min_idx[0], min_idx[1]
            diff = np.abs(dist - radius)
            rad_idx = np.unravel_index(np.argmin(diff), diff.shape)
            x_rad, y_rad = rad_idx[0], rad_idx[1]
            cent_rad = int(np.sqrt((x_rad - x_cent) ** 2 + (y_rad - y_cent) ** 2))
            cent = [int(x_cent - 1.5 * cent_rad), int(y_cent - 1.5 * cent_rad)]

            def _get_all_points(
                *,
                pts_lst_array: list[int],
                rng: np.random.Generator,
                n: int = 3,
                rad: float = 0.2,
                edgy: float = 0.5,
                scale: float = 1,
            ) -> np.ndarray:
                """Helper function to get all random points within Bézier curve."""
                pts_lst = [pts_lst_array[0:2]]
                enclosed_points = np.empty((0, 2), dtype=np.int64)
                for c in pts_lst:
                    points = (
                        WaterDropletPerturber.get_random_points_within_min_dist(
                            rng=rng,
                            n=n,
                            scale=scale,
                        )
                        + c[0:2]
                    )
                    x, y = WaterDropletPerturber.get_bezier_curve(
                        points=points,
                        rad=rad,
                        edgy=edgy,
                    )
                    curve_points = np.column_stack((x, y))
                    # Compute bounding box from curve points
                    xmin, ymin = curve_points.min(axis=0)
                    xmax, ymax = curve_points.max(axis=0)
                    grid_x, grid_y = np.mgrid[xmin:xmax:150j, ymin:ymax:150j]
                    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
                    # Use Numba-accelerated point-in-polygon test
                    inside_mask = _points_in_polygon(points=grid_points, polygon=curve_points)
                    enclosed_points = grid_points[inside_mask].astype(np.int64)

                return enclosed_points

            # Draw a Bézier shape centered at the center of the sphere and
            # find all the pixels that fall within the Bézier shape
            all_points = _get_all_points(
                pts_lst_array=cent,
                rng=self.rng,
                rad=0.6,
                scale=2 * cent_rad,
            )
            # Vectorized bounds check and assignment
            if all_points.size > 0:
                valid_mask = (
                    (all_points[:, 0] >= 0)
                    & (all_points[:, 0] < q.shape[0])
                    & (all_points[:, 1] >= 0)
                    & (all_points[:, 1] < q.shape[1])
                )
                valid_points = all_points[valid_mask]
                q[valid_points[:, 0], valid_points[:, 1]] = i
        return q

    def render(self, image: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Rendering the image with the Water Droplet effect.

        Args:
            image: Input Image.

        Returns:
            Image rendered with Water Droplet effect and an image mask.
        """
        h, w = image.shape[:2]

        # Initialize the intrinsic parameters of the camera
        # These defaults were based on what empirically looked most realistic
        # If using a particular camera, customization may be beneficial
        intrinsic = np.zeros((3, 3), dtype=np.float64)

        # Focal lengths
        intrinsic[0, 0] = self.f_x  # fx
        intrinsic[1, 1] = self.f_y  # fy

        # Offset of camera
        intrinsic[0, 2] = w / 2  # u0
        intrinsic[1, 2] = h / 2  # v0

        gls = self._to_glass(
            x=w,
            y=h,
            psi=self.psi,
            intrinsic=intrinsic,
            M=self.M,
        )

        # Initiate mask for blurring purposes
        mask = np.zeros((h, w), dtype=np.uint8)

        # Create random spherical droplets
        self._get_sphere_raindrop(width=w, height=h, gls=gls)

        # Create the matrix that determines what points are in the droplets
        # Also initializes the function that perturbs the spheres
        q = self._in_sphere_raindrop(gls)

        rain_image = copy.deepcopy(image)
        for idx in np.unique(q):
            if idx != -1:
                idx_int = int(idx)
                idxs = np.where(q == idx)
                xs = idxs[0].astype(np.int64)
                ys = idxs[1].astype(np.int64)

                if xs.size == 0:
                    continue

                # Vectorized computation of refraction mapping for all pixels in this droplet
                u_coords, v_coords = _compute_refraction_mapping(
                    xs=xs,
                    ys=ys,
                    gls=gls,
                    normal=self.normal,
                    n_air=self.n_air,
                    n_water=self.n_water,
                    M=self.M,
                    B=self.B,
                    center=self.centers[idx_int],
                    radius=self.radius[idx_int],
                    intrinsic=intrinsic,
                )

                # Vectorized bounds clamping
                u_coords = np.clip(u_coords, 0, w - 1).astype(np.int64)
                v_coords = np.clip(v_coords, 0, h - 1).astype(np.int64)

                # Vectorized image and mask update
                dest_y = ys - 1
                dest_x = xs - 1

                # Ensure destination indices are valid
                valid_mask = (dest_y >= 0) & (dest_y < h) & (dest_x >= 0) & (dest_x < w)
                dest_y = dest_y[valid_mask]
                dest_x = dest_x[valid_mask]
                u_coords = u_coords[valid_mask]
                v_coords = v_coords[valid_mask]

                rain_image[dest_y, dest_x] = image[v_coords, u_coords]
                mask[dest_y, dest_x] = 255

        return rain_image, mask

    def blur(
        self,
        *,
        image: np.ndarray[Any, Any],
        rain_image: np.ndarray[Any, Any],
        mask: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Blur the area within the boundaries of the droplets.

        Args:
            image (np.ndarray):
                Input Image.
            rain_image (np.ndarray):
                Image rendered with Water droplet effect.
            mask (np.ndarray):
                Image mask.

        Returns:
            Output of blur operation applied on the `rain_image`.
        """
        blur_image = copy.deepcopy(rain_image)
        _, w = image.shape[:2]
        blur_back = copy.deepcopy(rain_image)
        blur_values = [w / 40, w / 60]
        blur_values_adj = [int(np.floor(val / 2) * 2 + 1) for val in blur_values]

        blur_image = self._apply_gaussian(image=blur_image, sigma=w / 150, ksize=blur_values_adj[0])

        # Blur the background of the image using the desired blur strength
        blur_back = self._apply_gaussian(image=blur_back, sigma=1.5, ksize=7)

        blur_back = self.blur_strength * blur_back + (1 - self.blur_strength) * rain_image
        # Blur mask to help make the boundaries of the droplets appear "fuzzier"
        mask = self._apply_gaussian(image=mask, sigma=w / 125, ksize=blur_values_adj[1])

        blur_image[mask == 0] = blur_back[mask == 0]

        return blur_image

    def _apply_gaussian(self, *, image: np.ndarray[Any, Any], sigma: float, ksize: int) -> np.ndarray[Any, Any]:
        truncate = (ksize - 1) / 2 / sigma
        if image.ndim == 2:
            # Grayscale
            blurred = gaussian_filter(
                image.astype(np.float64),
                sigma=sigma,
                truncate=truncate,
                mode="grid-wrap",
            )
        else:
            # Color image – apply Gaussian to each channel independently
            blurred = np.empty_like(image, dtype=np.float64)
            for c in range(image.shape[2]):
                blurred[..., c] = gaussian_filter(
                    image[..., c].astype(np.float64),
                    sigma=sigma,
                    truncate=truncate,
                    mode="grid-wrap",
                )

        return blurred.astype(image.dtype)

    @override
    def perturb(
        self,
        *,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Applies the Water Droplet perturbation effect to the provided input image.

        Args:
            image:
                The image to be perturbed.
            boxes:
                Bounding boxes for source detections.
            kwargs:
                Additional perturbation keyword arguments (currently unused).

        Returns:
            The perturbed image and bounding boxes scaled to perturbed image shape.
        """
        image, boxes = super().perturb(image=image, boxes=boxes)

        # Reset RNG state when seed is provided to ensure deterministic results across multiple calls
        if self.seed is not None:
            self._initialize_derived_parameters()

        rain_image, mask = self.render(image=image)
        perturbed_image = self.blur(
            image=image,
            rain_image=rain_image,
            mask=mask,
        )

        if boxes:
            rescaled_boxes = self._rescale_boxes(
                boxes=boxes,
                orig_shape=image.shape,
                new_shape=perturbed_image.shape,
            )
            return perturbed_image.astype(np.uint8), rescaled_boxes

        return perturbed_image.astype(np.uint8), boxes

    @override
    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration of the WaterDropletPerturber instance."""
        cfg = super().get_config()

        cfg["size_range"] = self.size_range
        cfg["num_drops"] = self.num_drops
        cfg["blur_strength"] = self.blur_strength
        cfg["psi"] = self.psi
        cfg["n_air"] = self.n_air
        cfg["n_water"] = self.n_water
        cfg["f_x"] = self.f_x
        cfg["f_y"] = self.f_y
        cfg["seed"] = self.seed

        return cfg

    @classmethod
    @override
    def is_usable(cls) -> bool:
        """Returns true if the necessary dependencies (Scipy and Numba) are available."""
        return scipy_available and numba_available
