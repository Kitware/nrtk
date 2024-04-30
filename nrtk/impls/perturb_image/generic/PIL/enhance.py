from typing import Any, Dict, Protocol, Type, TYPE_CHECKING, runtime_checkable

from PIL import Image, ImageEnhance
from PIL.Image import Image as PILImage
import numpy as np

from nrtk.interfaces.perturb_image import PerturbImage


@runtime_checkable
class _Enhancement(Protocol):  # Used for type checking only  # pragma: no cover
    def __init__(self: '_Enhancement', image: PILImage):
        pass

    def enhance(self: '_Enhancement', factor: float) -> PILImage:
        pass


class _PILEnhancePerturber(PerturbImage):
    def __init__(
        self,
        factor: float = 1.0
    ):
        """
        :param factor: Enhancement factor.
        """
        if factor < 0.:
            raise ValueError(f"{type(self).__name__} invalid factor ({factor})."
                             f" Must be >= 0.0")

        self.factor = factor

    def _perturb(
        self,
        enhancement: Type[_Enhancement],
        image: np.ndarray,
        additional_params: Dict[str, Any] = {}
    ) -> np.ndarray:
        """
        Call appropriate enhancement interface and perform any necessary data
        type conversion.

        :param enhancement: Ehancement to apply.
        :param image: Input image as a numpy array.

        :return: Peturbed image as numpy array, including matching shape and dtype.
        """

        dtype = image.dtype
        # PIL does not support RGB floating point images so we must do an
        # intermediary conversion
        if np.issubdtype(dtype, np.floating):
            image = (image * 255).astype(np.uint8)

        image_pil = Image.fromarray(image)
        image_enhanced = enhancement(image_pil).enhance(factor=self.factor)
        image_np = np.array(image_enhanced)

        # Convert back to floating point dtype if needed
        if np.issubdtype(dtype, np.floating):
            image_np = image.astype(dtype) / 255

        return image_np

    def get_config(self) -> Dict[str, Any]:
        return {
            "factor": self.factor
        }


class BrightnessPerturber(_PILEnhancePerturber):
    """
    Adjusts image stimulus brightness.
    """

    def perturb(
        self,
        image: np.ndarray,
        additional_params: Dict[str, Any] = {}
    ) -> np.ndarray:
        """
        Return image stimulus with adjusted brightness.
        """

        enhancement = ImageEnhance.Brightness
        if TYPE_CHECKING and not isinstance(enhancement, _Enhancement):  # pragma: no cover
            raise ValueError("enhancement does not conform to _Enhancement protocol")
        return self._perturb(enhancement=enhancement, image=image)


class ColorPerturber(_PILEnhancePerturber):
    """
    Adjusts image stimulus color balance.
    """

    def perturb(
        self,
        image: np.ndarray,
        additional_params: Dict[str, Any] = {}
    ) -> np.ndarray:
        """
        Return image stimulus with adjusted color balance.
        """

        enhancement = ImageEnhance.Color
        if TYPE_CHECKING and not isinstance(enhancement, _Enhancement):  # pragma: no cover
            raise ValueError("enhancement does not conform to _Enhancement protocol")
        return self._perturb(enhancement=enhancement, image=image)


class ContrastPerturber(_PILEnhancePerturber):
    """
    Adjusts image stimulus contrast.
    """

    def perturb(
        self,
        image: np.ndarray,
        additional_params: Dict[str, Any] = {}
    ) -> np.ndarray:
        """
        Return image stimulus with adjusted contrast.
        """

        enhancement = ImageEnhance.Contrast
        if TYPE_CHECKING and not isinstance(enhancement, _Enhancement):  # pragma: no cover
            raise ValueError("enhancement does not conform to _Enhancement protocol")
        return self._perturb(enhancement=enhancement, image=image)


class SharpnessPerturber(_PILEnhancePerturber):
    """
    Adjusts image stimulus sharpness.
    """

    def __init__(
        self,
        factor: float = 1.0
    ):
        """
        :param rng: Enhancement factor.
        """
        if factor < 0. or factor > 2.0:
            raise ValueError(f"{type(self).__name__} invalid sharpness factor ({factor})."
                             f" Must be in [0.0, 2.0]")

        super().__init__(factor=factor)

    def perturb(
        self,
        image: np.ndarray,
        additional_params: Dict[str, Any] = {}
    ) -> np.ndarray:
        """
        Return image stimulus with adjusted sharpness.
        """

        enhancement = ImageEnhance.Sharpness
        if TYPE_CHECKING and not isinstance(enhancement, _Enhancement):  # pragma: no cover
            raise ValueError("enhancement does not conform to _Enhancement protocol")
        return self._perturb(enhancement=enhancement, image=image)
