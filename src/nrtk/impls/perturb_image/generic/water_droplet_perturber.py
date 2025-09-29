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
    WaterDropletPerturber: Implements the physics-based, photorealistic
    water/rain droplet model, utilizing Scipy, Shapely, and GeoPandas functionalities.

Dependencies:
    - Scipy for image processing.
    - Shapely and GeoPandas for Curve generation related operations.
    - nrtk.interfaces.perturb_image.PerturbImage as the base interface for image perturbation.

Example usage:
    size_range = (0.0, 1.0)
    num_drops = 20
    perturber = WaterDropletPerturber(size_range=size_range, num_drops=num_drops)
    perturbed_image, boxes = perturber.perturb(image, boxes)

Notes:
    - The boxes returned from `perturb` are identical to the boxes passed in.
"""

from __future__ import annotations

__all__ = ["WaterDropletPerturber"]

import copy
import math
from collections.abc import Hashable, Iterable, Sequence
from typing import Any

from typing_extensions import override

from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.utils._exceptions import WaterDropletImportError
from nrtk.utils._import_guard import import_guard

scipy_available: bool = import_guard("scipy", WaterDropletImportError, ["special", "ndimage"])
shapely_available: bool = import_guard("shapely", WaterDropletImportError, ["geometry"])
geopandas_available: bool = import_guard("geopandas", WaterDropletImportError)
import geopandas  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.special import binom  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402
from smqtk_image_io.bbox import AxisAlignedBoundingBox  # noqa: E402


class Bezier:
    """Class that computes the Bezier curve based on the segment information.

    Each curve is made of a series of segments that are initialized by the
    input points, angles, target radius and number of points needed for the
    Bezier interpolation.
    """

    def __init__(
        self,
        p1: np.ndarray[Any, Any],
        p2: np.ndarray[Any, Any],
        angle1: float,
        angle2: float,
        r: float = 0.3,
        num_points: int = 100,
    ) -> None:
        """Define segment parameters - points, angles, radius."""
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

        def _bernstein(n: int, k: int, t: np.ndarray) -> np.ndarray:
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
        seed (int):
            Random seed for reproducibility.
    """

    def __init__(
        self,
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
                Random seed for reproducibility.

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
            seed = None

        Raises:
            :raises ImportError: If Scipy, Shapely or GeoPandas is not found,
            install via `pip install nrtk[waterdroplet]`.
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
        self.M = self.rng.integers(20, 40)

        # Background plane which is B centimeters from the camera
        # and lies beyond the glass plane (value range chosen from source paper)
        self.B = self.rng.integers(800, 1500)

        self.normal: np.ndarray[Any, Any] = np.array([0.0, -1.0 * np.cos(self.psi), np.sin(self.psi)])

        self.g_centers = list()
        self.g_radius = list()
        self.centers = list()
        self.radius = list()

    @staticmethod
    def ccw_sort(points: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Sorts points in counterclockwise order around a center point."""
        # Subtract original point from center point (position obtained
        # by calculating the mean)
        points128 = points.astype(np.float128)
        d = points128 - np.mean(points128, axis=0)
        # Use atan2 to determine the angle taking into account the correct quadrant
        s = np.arctan2(d[:, 0], d[:, 1])
        # Return the sorted array of points.
        return points[np.argsort(s), :]

    @staticmethod
    def get_bezier_curve(
        points: np.ndarray[Any, Any],
        rad: float = 0.2,
        edgy: float = 0.0,
        tol: float = 1e-8,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Given an array of *points*, create a curve through those points.

        *rad* is a number between 0 and 1 to steer the distance of
            control points.
        *edgy* is a parameter which controls how "edgy" the curve is,
            edgy=0 is smoothest.
        *tol* is a parameter which controls the tolerance used when
            comparing angles. Default is 1e-8.
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

        def _get_curve(points: np.ndarray, r: float) -> np.ndarray:
            """Get the segments and curve data."""
            segments = list()
            for i in range(len(points) - 1):
                seg = Bezier(
                    points[i, :2],
                    points[i + 1, :2],
                    points[i, 2],
                    points[i + 1, 2],
                    r,
                )
                segments.append(seg.get_curve())

            return np.concatenate(segments)

        c = _get_curve(points=points, r=rad)
        x, y = c.T
        return x, y

    @staticmethod
    def get_random_points_within_min_dist(
        rng: np.random.Generator,
        n: int = 5,
        scale: float = 0.8,
        min_dst: float | None = None,
        recursive: int = 0,
    ) -> np.ndarray[Any, Any]:
        """Function to create *n* random points in the unit square, which are *min_dst* apart, then scale them."""
        min_dst = min_dst or 0.7 / n
        points = rng.random((n, 2))
        d = np.sqrt(np.sum(np.diff(WaterDropletPerturber.ccw_sort(points), axis=0), axis=1) ** 2)
        if np.all(d >= min_dst) or recursive >= 200:
            return points * scale
        return WaterDropletPerturber.get_random_points_within_min_dist(
            rng,
            n=n,
            scale=scale,
            min_dst=min_dst,
            recursive=recursive + 1,
        )

    def _to_glass(
        self,
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
            :return np.ndarray: Glass plane (3D) coordinate system matrix.
        """
        xx, yy = np.meshgrid(np.arange(x), np.arange(y), indexing="ij")
        w = M * np.tan(psi) / (np.tan(psi) - (yy - intrinsic[1, 2]) / intrinsic[1, 1])
        u = w * (xx - intrinsic[0, 2]) / intrinsic[0, 0]
        v = w * (yy - intrinsic[1, 2]) / intrinsic[1, 1]
        return np.dstack((u, v, w)).reshape((x, y, 3))

    def _get_sphere_raindrop(self, width: int, height: int, gls: np.ndarray[Any, Any]) -> None:  # noqa: C901
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

        def __random_tau() -> int:
            """Determines angle between tangent and glass plane."""
            return math.floor(self.rng.uniform(30, 45))

        def __random_loc() -> float:
            """Determine random multiplier value that is applied to the water droplet size computation."""
            return self.rng.uniform(self.size_range[0], self.size_range[1])

        def __w_in_plane(u: int, v: int) -> int:
            """Estimate the "depth" value of a pixel in the coordinate system of the glass plane."""
            return (self.normal[2] * self.M - self.normal[0] * u - self.normal[1] * v) / self.normal[2]

        def __remove_overlapping_drops() -> None:  # noqa: C901
            """Remove overlapping droplet spheres."""
            indices_to_remove = list()
            for i in range(len(self.g_centers)):
                center = self.g_centers[i]
                radius = self.g_radius[i]
                for j in range(len(self.g_centers)):
                    if i >= j:
                        continue
                    center_other = self.g_centers[j]
                    radius_other = self.g_radius[j]
                    distance_between_centers = np.linalg.norm(center - center_other)
                    sum_radii = radius + radius_other
                    if sum_radii > distance_between_centers:
                        indices_to_remove.append(i)
                        # we don't need to identify an overlapping drop more than once
                        break
            indices_to_remove.sort(reverse=True)

            for index in indices_to_remove:
                self.g_centers.pop(index)
                self.g_radius.pop(index)
                self.centers.pop(index)
                self.radius.pop(index)

        for _ in range(self.num_drops):
            u = left_bottom[0] + (right_bottom[0] - left_bottom[0]) * self.rng.random()
            v = left_upper[1] + (right_bottom[1] - left_upper[1]) * self.rng.random()
            w = __w_in_plane(u, v)

            # Convert the angle between tangent and glass plane from degrees to radians
            tau = __random_tau() / 180 * np.pi

            # Water droplet size computation
            glass_r = 0.1 + (width // 500) * __random_loc()

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
        __remove_overlapping_drops()

    def _in_sphere_raindrop(self, gls: np.ndarray) -> np.ndarray:  # noqa: C901
        """Helper function for rendering.

        Determine if a given pixel (x, y) is inside any simulated raindrop on
        the windshield. If yes, == index of the raindrop, if no, == -1.

        Args:
            gls: Glass (3D) coordinate system mapping matrix.

        Returns:
            :return np.ndarray: Truth mask of valid pixels.
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
            x_cent = np.where(dist == np.min(dist))[0][0]
            y_cent = np.where(dist == np.min(dist))[1][0]
            diff = np.abs(dist - radius)
            rad_loc = np.where(diff == np.min(diff))
            x_rad, y_rad = rad_loc
            cent_rad = int(np.sqrt((x_rad - x_cent) ** 2 + (y_rad - y_cent) ** 2))
            cent = [int(x_cent - 1.5 * cent_rad), int(y_cent - 1.5 * cent_rad)]

            def __get_all_points(
                pts_lst_array: list[int],
                rng: np.random.Generator,
                n: int = 3,
                rad: float = 0.2,
                edgy: float = 0.5,
                scale: float = 1,
            ) -> list[np.ndarray]:
                """Helper function to get all random points within Bézier curve."""
                pts_lst = [pts_lst_array[0:2]]
                enclosed_points = list()
                for c in pts_lst:
                    points = (
                        WaterDropletPerturber.get_random_points_within_min_dist(
                            rng,
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
                    polygon = Polygon(curve_points)
                    xmin, ymin, xmax, ymax = polygon.bounds
                    grid_x, grid_y = np.mgrid[xmin:xmax:150j, ymin:ymax:150j]
                    grid_x, grid_y = grid_x.flatten(), grid_y.flatten()
                    points_gs = geopandas.GeoSeries(geopandas.points_from_xy(grid_x, grid_y))
                    enclosed_points = [
                        np.asarray([int(grid_x[i]), int(grid_y[i])])
                        for i, val in enumerate(polygon.contains(points_gs))
                        if val
                    ]

                return enclosed_points

            # Draw a Bézier shape centered at the center of the sphere and
            # find all the pixels that fall within the Bézier shape
            all_points = __get_all_points(
                cent,
                self.rng,
                rad=0.6,
                scale=2 * cent_rad,
            )
            for point in all_points:
                if point[0] >= q.shape[0] or point[1] >= q.shape[1]:
                    continue
                q[point[0], point[1]] = i
        return q

    def _to_sphere_section_env(self, x: int, y: int, idx: int, intrinsic: np.ndarray, gls: np.ndarray) -> np.ndarray:
        """Helper function for rendering.

        Calculate where a point (x, y) would map to on a sphere's surface,
        considering the effects of refraction (bending of light as it passes through
        different media).

        How it works:
        - Converts the pixel to a 3D point in the glass coordinate system.
        - Applies refraction and other transformations to find the equivalent point on the sphere's surface.
        - Maps this point back to the image plane and returns the adjusted pixel coordinates.

        Args:
            x:
                x-value from 2D coordinate system.
            y:
                x-value from 2D coordinate system.
            idx:
                Index for idx-th points from a list of points.
            intrinsic:
                Intrinsic (Camera) parameters matrix.
            gls:
                Glass (3D) coordinate system mapping matrix.

        Returns:
            :return np.ndarray: 3D point coordinates of water droplet (spherical model).
        """
        p_g = gls[x, y]

        alpha = np.arccos(np.dot(p_g, self.normal) / np.linalg.norm(p_g))
        beta = np.arcsin(self.n_air * np.sin(alpha) / self.n_water)
        o_g = (self.normal[2] * self.M / np.dot(self.normal, self.normal)) * self.normal

        po = p_g - o_g
        po = po / np.linalg.norm(po)
        i_1 = self.normal + np.tan(beta) * po
        i_1 = i_1 / np.linalg.norm(i_1)

        oc = p_g - self.centers[idx]
        tmp = np.dot(i_1, oc)
        d = -(tmp) + np.sqrt(abs(tmp**2 - np.dot(oc, oc) + self.radius[idx] ** 2))
        p_w = p_g + d * i_1

        normal_w = p_w - self.centers[idx]
        normal_w = normal_w / np.linalg.norm(normal_w)

        d = (np.dot(p_w, normal_w) - np.dot(normal_w, p_g)) / np.dot(normal_w, normal_w)
        p_a = p_w - (d * normal_w + p_g)
        p_a = p_a / np.linalg.norm(p_a)

        eta = np.arccos(np.dot(normal_w, p_w - p_g) / np.linalg.norm(p_w - p_g))

        # The angle between the normal to the spherical surface at the point of
        # incidence and the optical axis.
        gamma = np.arcsin(self.n_air / self.n_water)
        if eta >= gamma:
            eta = gamma - 0.2

        theta = np.arcsin(self.n_water * np.sin(eta) / self.n_air)
        i_2 = normal_w + np.tan(theta) * p_a
        p_e = p_w + ((self.B - p_w[2]) / i_2[2]) * i_2
        p_i = np.dot(intrinsic, np.transpose(p_e)) / self.B
        return np.round(p_i)

    def render(self, image: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:  # noqa: C901
        """Rendering the image with the Water Droplet effect.

        Args:
            image: Input Image.

        Returns:
            :return np.ndarray: Image rendered with Water Droplet effect.
            :return np.ndarray: Image mask.
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
        self._get_sphere_raindrop(w, h, gls)

        # Create the matrix that determines what points are in the droplets
        # Also initializes the function that perturbs the spheres
        q = self._in_sphere_raindrop(gls)

        rain_image = copy.deepcopy(image)
        for idx in np.unique(q):
            if idx != -1:
                idxs = np.where(q == idx)
                for _, (x, y) in enumerate(zip(idxs[0], idxs[1], strict=False)):
                    # Translate refractive distortions to the "surface" of the droplet
                    p = self._to_sphere_section_env(x, y, int(idx), intrinsic, gls)
                    u = p[0]
                    v = p[1]
                    # Conditions to keep the droplets within the bounds of the image
                    if u >= w:
                        u = w - 1
                    elif u < 0:
                        u = 0
                    if v >= h:
                        v = h - 1
                    elif v < 0:
                        v = 0

                    # Add effects to image and draw boundaries in image mask
                    rain_image[y - 1, x - 1] = image[int(v), int(u)]
                    mask[y - 1, x - 1] = 255
        return rain_image, mask

    def blur(
        self,
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
            np.ndarray: Output of blur operation applied on the `rain_image`.
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

    def _apply_gaussian(self, image: np.ndarray[Any, Any], sigma: float, ksize: int) -> np.ndarray[Any, Any]:
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
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        **additional_params: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Applies the Water Droplet perturbation effect to the provided input image.

        Args:
            image:
                The image to be perturbed.
            boxes:
                Bounding boxes for source detections.
            additional_params:
                Additional perturbation keyword arguments (currently unused).

        Returns:
            :return tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
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

    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration of the WaterDropletPerturber instance.

        Returns:
            :return dict[str, Any]: Configuration dictionary with current settings.
        """
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
    def is_usable(cls) -> bool:
        """Checks if the necessary dependencies (Scipy, Shapely and GeoPandas) are available.

        Returns:
            :return bool: True if Scipy, Shapely and GeoPandas are available.
        """
        return scipy_available and shapely_available and geopandas_available
