"""Defines RandomScalePerturber to apply Albumentations' `Scale` transformation to input images.

Classes:
    RandomScalePerturber: A perturbation class for applying the `Scale` transformation from Albumentations.

Dependencies:
    - albumentations: For the underlying perturbations
    - numpy: For numerical operations and random number generation.
    - smqtk_image_io.AxisAlignedBoundingBox: For handling and adjusting bounding boxes.
    - nrtk.interfaces.perturb_image.PerturbImage: Base class for perturbation algorithms.
    - nrtk.impls.perturb_image.generic.albumentations_perturber: Base implementation for Albumentations perturbers.
"""

from __future__ import annotations

from typing import Any

from typing_extensions import override

from nrtk.impls.perturb_image.generic.albumentations_perturber import AlbumentationsPerturber
from nrtk.utils._exceptions import AlbumentationsImportError, OpenCVImportError
from nrtk.utils._import_guard import import_guard

cv2_available: bool = import_guard("cv2", OpenCVImportError)

import cv2  # noqa: E402


class RandomScalePerturber(AlbumentationsPerturber):
    """RandomScalePerturber applies a Scale transformation from Albumentations.

    Attributes:
        limit (float | tuple[float, float]):
            Range of scaling factors. A single float value is equivalent to a tuple of (-value, value).
            One will then be added to each value. Therefore, a given value of (-0.2, 0.1) corresponds
            to scaling the image between 0.8x and 1.1x, and a given value of 0.5 corresponds to scaling
            the image between 0.5x and 1.5x.
        interpolation (int)
            OpenCV flag indicating the interpolation algorithm.
        probability (float):
            Probability of applying the scale transformation.
        seed (int):
            An optional seed for reproducible results.

    Methods:
        perturb:
            Applies the specified perturbation to an input image.
        __call__:
            Calls the perturb method with the given input image.
        get_config:
            Returns the current configuration of the RandomRotationPerturber instance.
    """

    def __init__(  # noqa: C901
        self,
        limit: float | tuple[float, float] = 0.0,
        interpolation: int = cv2.INTER_LINEAR,
        probability: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """RandomScalePerturber applies a random scale perturbation to an input image.

        It ensures that bounding boxes are adjusted correctly to reflect the translated
        image coordinates.

        Args:
            limit:
                Range of scaling factors. A single float value is equivalent to a tuple of (-value, value).
                One will then be added to each value. Therefore, a given value of (-0.2, 0.1) corresponds
                to scaling the image between 0.8x and 1.1x, and a given value of 0.5 corresponds to scaling
                the image between 0.5x and 1.5x.
            interpolation:
                OpenCV flag indicating the interpolation algorithm.
            probability:
                Probability of applying the scale transformation.
            seed:
                An optional seed for reproducible results.
        """
        if not self.is_usable():
            raise AlbumentationsImportError

        self._allowed_interpolations = {
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_LINEAR_EXACT,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        }

        if isinstance(limit, tuple):
            lower, upper = limit
        else:
            lower = -limit
            upper = limit

        if not 0.0 <= probability <= 1.0:
            raise ValueError("Scale probability must be between 0.0 and 1.0 inclusive.")
        if lower > upper:
            raise ValueError("Lower scale limit must be less than or equal to upper limit.")
        if lower <= -1.0:
            raise ValueError("Lower scale limit must be greater than -1.0.")
        if interpolation not in self._allowed_interpolations:
            raise ValueError("Interpolation value not supported.")

        super().__init__(
            perturber="RandomScale",
            parameters={"scale_limit": limit, "interpolation": interpolation, "p": probability},
            seed=seed,
        )

        self.limit = limit
        self.interpolation = interpolation
        self.probability = probability

    @override
    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration of the RandomRotationPerturber instance.

        Returns:
            :return dict[str, Any]: Configuration dictionary with current settings.
        """
        cfg = dict()
        cfg["limit"] = self.limit
        cfg["interpolation"] = self.interpolation
        cfg["probability"] = self.probability
        cfg["seed"] = self.seed
        return cfg
