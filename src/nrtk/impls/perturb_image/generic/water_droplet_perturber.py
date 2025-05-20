"""
# This file includes code originally posted on Stack Overflow
# (https://stackoverflow.com/posts/50751932/revisions) and is licensed under the
# Creative Commons Attribution-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-sa/4.0/
#
# © 2025 Kitware, Inc

WaterDropletPerturber Implementation based on the work from these source papers:
(1)
https://openaccess.thecvf.com/content_ICCVW_2019/papers/PBDL/Hao_Learning_From_Synthetic_Photorealistic_Raindrop_for_Single_Image_Raindrop_Removal_ICCVW_2019_paper.pdf
(2) https://www.giss.nasa.gov/pubs/abs/er05000f.html

For additional research regarding Water Droplet modeling, please refer to this paper:
https://www.cvlibs.net/publications/Roser2010ACCVWORK.pdf

Classes:
    WaterDropletPerturber: Implements the physics-based, photorealistic
    water/rain droplet model, utilizing OpenCV and Shapely functionalities.

Dependencies:
    - OpenCV for image processing.
    - Shapely for Curve generation related operations.
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

import copy
import math
from collections.abc import Hashable, Iterable, Sequence
from typing import Any

from typing_extensions import override

try:
    # Multiple type ignores added for pyright's handling of guarded imports
    import cv2

    cv2_available = True
except ImportError:  # pragma: no cover
    cv2_available = False

try:
    # Guarded import check for utility function usage.
    from shapely.geometry import Point, Polygon

    shapely_available = True
except ImportError:  # pragma: no cover
    shapely_available = False

try:
    # Guarded import check for utility function usage.
    import scipy  # noqa: F401

    scipy_available = True
except ImportError:  # pragma: no cover
    scipy_available = False


import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.utils._exceptions import WaterDropletImportError

if shapely_available and scipy_available:
    from nrtk.impls.perturb_image.generic.utils.water_droplet_perturber_utils import (
        get_bezier_curve,
        get_random_points_within_min_dist,
    )


class WaterDropletPerturber(PerturbImage):
    """
    Implements the physics-based, photorealistic water/rain droplet model.

    The `WaterDropletPerturber` class simulates the effects of rain/water droplets
    on an image similar to the rain drops on a window, car-windshield, etc. The equations
    defined for this model are based on the dynamics, geometry and photometry of a water/rain droplet.

    Attributes:
        size_range (Sequence[float]): Range of size multiplier values used for computing
            the size of the water droplet.
        num_drops (int): Target number of water droplets.
        blur_strength (float): Strength of Gaussian blur operation.
        psi (float): Angle between the camera line-of-sight and glass plane (radians).
        n_air (float): Density of air.
        n_water (float): Density of water.
        f_x (int): Camera focal length in x direction (cm).
        f_y (int): Camera focal length in y direction (cm).
        seed (int): Random seed for reproducibility.
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
        seed: int | None = None,
        box_alignment_mode: str = "extent",
    ) -> None:
        """Initializes the WaterDropletPerturber.

        :param size_range: Range of size multiplier values used for computing
            the size of the water droplet.
        :param num_drops: Target number of water droplets.
        :param blur_strength: Strength of Gaussian blur operation.
        :param psi: Angle between the line-of-sight and glass plane (radians).
        :param n_air: Density of air.
        :param n_water: Density of water.
        :param f_x: Camera focal length in x direction (cm).
        :param f_y: Camera focal length in y direction (cm).
        :param seed: Random seed for reproducibility.
        :param box_alignment_mode: Mode for how to handle how bounding boxes change.
            Should be one of the following options:
                extent: a new axis-aligned bounding box that encases the transformed misaligned box
                extant: a new axis-aligned bounding box that is encased inside the transformed misaligned box
                median: median between extent and extant
            Default value is extent

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

        :raises: ImportError if OpenCV, Scipy or Shapely is not found, install via
        `pip install nrtk[waterdroplet,graphics]` or `pip install nrtk[waterdroplet,headless]`.
        """
        if not self.is_usable():
            raise WaterDropletImportError
        super().__init__(box_alignment_mode=box_alignment_mode)
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
        """Derived Parameters"""
        self.rng = np.random.default_rng(self.seed)
        # Glass plane at M centimeters ahead of the camera (value range chosen from source paper)
        self.M = self.rng.integers(20, 40)

        # Background plane which is B centimeters from the camera
        # and lies beyond the glass plane (value range chosen from source paper)
        self.B = self.rng.integers(800, 1500)

        self.normal = np.array([0.0, -1.0 * np.cos(self.psi), np.sin(self.psi)])

        self.g_centers = list()
        self.g_radius = list()
        self.centers = list()
        self.radius = list()

    def _to_glass(
        self,
        x: int,
        y: int,
        psi: float,
        M: int,  # noqa: N803
        intrinsic: np.ndarray,
    ) -> np.ndarray:
        """
        Convert 2D pixel coordinates from an image (x, y) into a 3D point in the glass coordinate system.

        Args:
            x (int): X-coordinate value in the image plane.
            y (int): Y-coordinate value in the image plane.
            psi (float): Angle between the camera line-of-sight and glass plane (radians).
            M (int): Glass plane at M centimeters ahead of the camera.
            intrinsic: Intrinsic (Camera) parameters matrix.

        Returns:
            np.ndarray: Glass plane (3D) coordinate system matrix.
        """
        xx, yy = np.meshgrid(np.arange(x), np.arange(y), indexing="ij")
        w = M * np.tan(psi) / (np.tan(psi) - (yy - intrinsic[1, 2]) / intrinsic[1, 1])
        u = w * (xx - intrinsic[0, 2]) / intrinsic[0, 0]
        v = w * (yy - intrinsic[1, 2]) / intrinsic[1, 1]
        return np.dstack((u, v, w)).reshape((x, y, 3))

    def _get_sphere_raindrop(self, width: int, height: int, gls: np.ndarray) -> None:  # noqa: C901
        """
        Simulate and store information about raindrops on the windshield.

        How it works:
        - Defines random parameters for new raindrops, like size and position.
        - Generates raindrop spheres and their attributes (e.g., center, radius).
        - Stores this data for later use in the simulation.

        Args:
            width (int): Input image width.
            height (int): Input image height.
            gls (np.ndarray): Glass (3D) coordinate system mapping matrix.
        """
        self.g_centers = list()
        self.g_radius = list()
        self.centers = list()
        self.radius = list()

        left_upper = gls[0][0]
        left_bottom = gls[0][height - 1]
        right_bottom = gls[width - 1][height - 1]

        def __random_tau() -> int:
            """Determines angle between tangent and glass plane"""
            return math.floor(self.rng.uniform(30, 45))

        def __random_loc() -> float:
            """Determine random multiplier value that is applied to the water droplet size computation"""
            return self.rng.uniform(self.size_range[0], self.size_range[1])

        def __w_in_plane(u: int, v: int) -> int:
            """Estimate the "depth" value of a pixel in the coordinate system of the glass plane"""
            return (self.normal[2] * self.M - self.normal[0] * u - self.normal[1] * v) / self.normal[2]

        def __remove_overlapping_drops() -> None:  # noqa: C901
            """Remove overlapping droplet spheres"""
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
        """
        Determine if a given pixel (x, y) is inside any simulated raindrop on
        the windshield. If yes, == index of the raindrop, if no, == -1.

        Args:
            gls (np.ndarray): Glass (3D) coordinate system mapping matrix.

        Returns:
            np.ndarray: Truth mask of valid pixels.
        """
        p = gls
        q = np.ones(p.shape[:2]) * -1
        for i, (center, radius) in enumerate(zip(self.g_centers, self.g_radius)):
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
                """
                This internal function gets random points, draws the Bézier curve,
                and does a grid search to determine which points are within the curve.
                """
                pts_lst = [pts_lst_array[0:2]]
                enclosed_points = list()
                for c in pts_lst:
                    points = (
                        get_random_points_within_min_dist(  # pyright: ignore [reportPossiblyUnboundVariable]
                            rng,
                            n=n,
                            scale=scale,
                        )
                        + c[0:2]
                    )
                    x, y = get_bezier_curve(  # pyright: ignore [reportPossiblyUnboundVariable]
                        points=points,
                        rad=rad,
                        edgy=edgy,
                    )
                    curve_points = np.column_stack((x, y))
                    polygon = Polygon(curve_points)  # pyright: ignore [reportPossiblyUnboundVariable]
                    xmin, ymin, xmax, ymax = polygon.bounds
                    grid_points = np.mgrid[xmin:xmax:150j, ymin:ymax:150j].reshape(2, -1).T
                    enclosed_points = [
                        np.array([int(p[0]), int(p[1])])
                        for p in grid_points
                        if polygon.contains(Point(p))  # pyright: ignore [reportPossiblyUnboundVariable]
                    ]

                return enclosed_points

            # Draw a Bézier shape centered at the center of the sphere and
            # find all the pixels that fall within the Bézier shape
            all_points = __get_all_points(  # pyright: ignore [reportPossiblyUnboundVariable]
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
        """
        Calculate where a point (x, y) would map to on a sphere's surface,
        considering the effects of refraction (bending of light as it passes through
        different media).

        How it works:
        - Converts the pixel to a 3D point in the glass coordinate system.
        - Applies refraction and other transformations to find the equivalent point on the sphere's surface.
        - Maps this point back to the image plane and returns the adjusted pixel coordinates.

        Args:
            x (int): x-value from 2D coordinate system.
            y (int): x-value from 2D coordinate system.
            idx (int): Index for idx-th points from a list of points.
            intrinsic: Intrinsic (Camera) parameters matrix.
            gls (np.ndarray): Glass (3D) coordinate system mapping matrix.

        Returns:
            np.ndarray: 3D point coordinates of water droplet (spherical model).
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
        gamma = np.arcsin(self.n_air / self.n_water)
        if eta >= gamma:
            eta = gamma - 0.2

        theta = np.arcsin(self.n_water * np.sin(eta) / self.n_air)
        i_2 = normal_w + np.tan(theta) * p_a
        p_e = p_w + ((self.B - p_w[2]) / i_2[2]) * i_2
        p_i = np.dot(intrinsic, np.transpose(p_e)) / self.B
        return np.round(p_i)

    def render(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # noqa: C901
        """
        Rendering the image with the Water Droplet effect.

        Args:
            image (np.ndarray): Input Image.

        Returns:
            np.ndarray: Image rendered with Water Droplet effect.
            np.ndarray: Image mask.
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
                for _, (x, y) in enumerate(zip(idxs[0], idxs[1])):
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

    def blur(self, image: np.ndarray, rain_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Blur the area within the boundaries of the droplets

        Args:
            image (np.ndarray): Input Image.
            rain_image (np.ndarray): Image rendered with Water droplet effect.
            mask (np.ndarray): Image mask.

        Returns:
            np.ndarray: Output of blur operation applied on the `rain_image`.
        """
        blur_image = copy.deepcopy(rain_image)
        _, w = image.shape[:2]
        blur_back = copy.deepcopy(rain_image)
        blur_values = [w / 40, w / 60]
        blur_values_adj = [int(np.floor(val / 2) * 2 + 1) for val in blur_values]
        blur_image = cv2.GaussianBlur(  # pyright: ignore [reportPossiblyUnboundVariable]
            blur_image,
            (blur_values_adj[0], blur_values_adj[0]),
            w / 150,
        )

        # Blur the background of the image using the desired blur strength
        blur_back = cv2.GaussianBlur(blur_back, (7, 7), 1.5)  # pyright: ignore [reportPossiblyUnboundVariable]
        blur_back = cv2.addWeighted(  # pyright: ignore [reportPossiblyUnboundVariable]
            blur_back,
            self.blur_strength,
            rain_image,
            1 - self.blur_strength,
            0,
        )
        # Blur mask to help make the boundaries of the droplets appear "fuzzier"
        mask = cv2.GaussianBlur(  # pyright: ignore [reportPossiblyUnboundVariable]
            mask,
            (blur_values_adj[1], blur_values_adj[1]),
            w / 125,
        )
        blur_image[mask == 0] = blur_back[mask == 0]

        return blur_image

    @override
    def perturb(
        self,
        image: np.ndarray,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        additional_params: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """
        Applies the Water Droplet perturbation effect to the provided input image.

        Args:
            image (np.ndarray): The image to be perturbed.
            boxes (Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]], optional): Bounding boxes
                for detections in input image
            additional_params (dict[str, Any], optional): Additional parameters, if applicable.

        Returns:
            np.ndarray: The perturbed image.
            Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]: Bounding boxes
                scaled to perturbed image shape.
        """
        image, boxes = super().perturb(image=image, boxes=boxes)

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
        """
        Returns the current configuration of the WaterDropletPerturber instance.

        Returns:
            dict[str, Any]: Configuration dictionary with current settings.
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
        """
        Checks if the necessary dependencies (OpenCV, Scipy and Shapely) are available.

        Returns:
            bool: True if OpenCV, Scipy and Shapely are available.
        """
        return cv2_available and scipy_available and shapely_available
