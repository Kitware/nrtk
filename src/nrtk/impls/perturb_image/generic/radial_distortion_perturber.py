"""Defines the RadialDistortionPerturber which applies radial distortion to input images.

The Radial distortion perturbation simulates lens distortion effects, such as
barrel or pincushion distortion, and is useful for augmenting datasets to improve
robustness to camera nonlinearities.

The distortion uses the following equations:
    x1 = x0 * (1 + k1*r^2 + k2*r^4 + k3*r^6)
    y1 = y0 * (1 + k1*r^2 + k2*r^4 + k3*r^6)
Where r is the distance from the image center and k1-k3 are distortion coefficients.

The class also adjusts associated axis-aligned bounding boxes to match the
distorted image, making it suitable for tasks involving labeled datasets.

Classes:
    RadialDistortionPerturber: A perturbation class for applying random radial
    distortion to images and updating their corresponding bounding boxes.

Dependencies:
    - numpy: For numerical operations and array manipulation.
    - smqtk_image_io.AxisAlignedBoundingBox: For handling and adjusting bounding boxes.
    - nrtk.interfaces.perturb_image.PerturbImage: Base class for image perturbation algorithms.
"""

from __future__ import annotations

__all__ = ["RadialDistortionPerturber"]

from collections.abc import Hashable, Iterable, Sequence
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.interfaces.perturb_image import PerturbImage


class RadialDistortionPerturber(PerturbImage):
    """RadialDistortionPerturber applies a radial distortion to an image.

    Attributes:
        k (Sequence[float]): A list of coefficients used to compute the radial distortion,
            should have exactly 3 values. Positive values represent a fisheye or barrel distortion,
            negative values represent a pincushion distortion.
        color_fill (Sequence[int]): Background color fill for RGB image.

    Methods:
        perturb:
            Applies the configured radial distortion to the input image and updates bounding boxes.
        __call__:
            Invokes the perturb method with the given input image and annotations.
        get_config:
            Returns the current configuration of the RadialDistortionPerturber instance.
    """

    def __init__(
        self,
        k: Sequence[float] = [0, 0, 0],
        color_fill: Sequence[int] | None = [0, 0, 0],
    ) -> None:
        """Applies a radial distortion to an image.

        Args:
            k (Sequence[float]): A list of coefficients used to compute the radial distortion.
            color_fill (Sequence[int]): Background color fill for RGB image.

        Raises:
            ValueError: Errors when k does not have exactly 3 values
        """
        super().__init__()
        self.k = k
        if len(k) != 3:
            raise ValueError("k must have exactly 3 values")
        self.color_fill: np.ndarray[Any, Any] = np.array(color_fill)

    def _radial_transform(
        self,
        x0: np.ndarray[Any, Any],
        y0: np.ndarray[Any, Any],
        w: float,
        h: float,
        k: Sequence[float],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Helper to transform coordinates using radial distortion.

        Args:
            x0 (np.ndarray[Any, Any]): Mesh grid of image x coordinates
            y0 (np.ndarray[Any, Any]): Mesh grid of image y coordinates
            w (float): Image width as a float
            h (float): Image height as a float
            k (Sequence[float]): Radial distortion coefficients

        Returns:
            np.ndarray[Any, Any]: Distorted x coordinate mesh
            np.ndarray[Any, Any]: Distorted y coordinate mesh
        """
        # Short-circuit when all k=0 to avoid calculation
        if all(ki == 0 for ki in k):
            return x0, y0

        # Normalize x0, y0 to range [-1, 1]
        x_norm = (x0 / (w - 1)) * 2 - 1
        y_norm = (y0 / (h - 1)) * 2 - 1

        # Radial distance (r^2, r^4, r^6)
        r2 = x_norm**2 + y_norm**2
        r4 = r2**2
        r6 = r2 * r4

        # Calculate distorted coordinates
        distortion = 1 + k[0] * r2 + k[1] * r4 + k[2] * r6
        x1 = x_norm * distortion
        y1 = y_norm * distortion

        # Convert back to pixel coordinates
        x1 = (((x1 + 1) / 2) * (w - 1)).astype(int)
        y1 = (((y1 + 1) / 2) * (h - 1)).astype(int)

        return x1, y1

    def perturb(
        self,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        **_: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Applies a radial distortion to an image and adjusts bounding boxes.

        Args:
            image:
                Input image as a numpy array of shape (H, W, C).
            boxes:
                List of bounding boxes in AxisAlignedBoundingBox format and their corresponding classes.
            additional_params:
                Additional perturbation keyword arguments (currently unused).

        Returns:
            np.ndarray[Any, Any]: Distorted image as numpy array
            Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]: Updated bounding boxes
        """
        super().perturb(image=image)

        # Get w, h and empty output image
        h, w = float(image.shape[0]), float(image.shape[1])
        out = np.full_like(image, self.color_fill.astype(image.dtype), dtype=image.dtype)

        # Get distorted coordinates
        x0, y0 = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        x1, y1 = self._radial_transform(x0, y0, w, h, self.k)

        # Valid index mask
        valid_mask = (x1 >= 0) & (x1 < image.shape[1]) & (y1 >= 0) & (y1 < image.shape[0])

        # Assign using correct shape and axis order
        out[y0[valid_mask], x0[valid_mask]] = image[y1[valid_mask], x1[valid_mask]]

        # Update bounding boxes
        if boxes:
            boxes = list(boxes)
            for i in range(len(boxes)):
                box, label = boxes[i]

                # Get corners
                min_x, min_y = box.min_vertex
                max_x, max_y = box.max_vertex
                x0 = np.array([min_x, max_x, max_x, min_x])
                y0 = np.array([min_y, min_y, max_y, max_y])

                # Transform corners
                x1, y1 = self._radial_transform(x0, y0, w, h, [-k for k in self.k])

                # New axis-aligned bounding box from distorted corners
                boxes[i] = (self._align_box(np.transpose([x1, y1])), label)

        return out.astype(np.uint8), boxes

    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration of the RadialDistortionPerturber instance.

        Returns:
            dict[str, Any]: Configuration dictionary with current settings.
        """
        cfg = super().get_config()
        cfg["k"] = self.k
        cfg["color_fill"] = self.color_fill
        return cfg
