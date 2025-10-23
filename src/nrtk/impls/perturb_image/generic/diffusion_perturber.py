"""Defines DiffusionPerturber, a PerturbImage implementation that uses diffusion models for prompt-based perturbations.

Classes:
    DiffusionPerturber: An implementation of the `PerturbImage` interface that applies diffusion-based
    perturbations to input images using pre-trained models and text prompts.

Dependencies:
    - numpy for handling image data arrays
    - torch for PyTorch functionality
    - diffusers for Stable Diffusion models
    - PIL for image processing
    - nrtk.interfaces for the `PerturbImage` interface
    - smqtk_image_io for bounding box handling

Example:
    diffusion_perturber = DiffusionPerturber(
        model_name="timbrooks/instruct-pix2pix",
        prompt="add rain to the image"
    )
    perturbed_image, perturbed_boxes = diffusion_perturber.perturb(input_image, boxes)

Note:
    This implementation uses the Instruct Pix2Pix model for prompt-based image transformations.
    The model is loaded on first use and cached for subsequent operations.
    Bounding boxes are not expected to be accurate after perturbation due to the generative nature of diffusion.
"""

from __future__ import annotations

__all__ = ["DiffusionPerturber"]

import warnings
from collections.abc import Hashable, Iterable
from typing import Any, cast

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.utils._exceptions import DiffusionImportError
from nrtk.utils._import_guard import import_guard

torch_available: bool = import_guard("torch", DiffusionImportError, fake_spec=True)
transformers_available: bool = import_guard("transformers", DiffusionImportError)
diffusion_available: bool = import_guard(
    "diffusers",
    DiffusionImportError,
    [
        "pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix",
        "schedulers.scheduling_euler_ancestral_discrete",
    ],
)
pillow_available: bool = import_guard("PIL", DiffusionImportError, ["Image"])
import torch  # noqa: E402
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix import (  # noqa: E402
    StableDiffusionInstructPix2PixPipeline,
)
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler  # noqa: E402
from PIL.Image import Image, Resampling, fromarray  # noqa: E402
from transformers import CLIPTextModel  # noqa: E402


class DiffusionPerturber(PerturbImage):
    """Diffusion-based implementation of the ``PerturbImage`` interface for prompt-guided perturbations.

    This class uses diffusion models (specifically the Instruct Pix2Pix model) to generate realistic
    perturbations on input images based on text prompts. The perturber can apply various effects
    and transformations guided by natural language descriptions.

    Args:
        model_name: Name of the pre-trained diffusion model from Hugging Face.
                   Default is "timbrooks/instruct-pix2pix".
        prompt: Text prompt describing the desired perturbation or transformation.
        seed: Random seed for reproducible perturbations. If None, perturbations will be non-deterministic.
        num_inference_steps: Number of denoising steps. Default is 50.
        text_guidance_scale: Guidance scale for text prompt. Default is 8.0.
        image_guidance_scale: Guidance scale for image conditioning. Default is 2.0.
        device: Device for computation, e.g., "cpu" or "cuda". If None, selects
                CUDA if available, otherwise CPU. Default is None.

    Note:
        The model is loaded lazily on first use and cached for subsequent operations.
        Images are automatically resized to be compatible with the diffusion model.
        Images will be resized to a minimum dimension of 256 pixels and dimensions
        divisible by 8 for optimal diffusion model performance.
        Device selection is automatic: CUDA is used if available, otherwise CPU.
    """

    def __init__(
        self,
        model_name: str = "timbrooks/instruct-pix2pix",
        prompt: str = "do not change the image",
        seed: int = 1,
        num_inference_steps: int = 50,
        text_guidance_scale: float = 8.0,
        image_guidance_scale: float = 2.0,
        device: str | None = None,
    ) -> None:
        """Initialize the DiffusionPerturber with configuration parameters.

        Args:
            model_name: Name of the pre-trained diffusion model. Default is "timbrooks/instruct-pix2pix".
            prompt: Text prompt describing the desired perturbation. Examples include
                "add rain to the image", "make it foggy", "add snow", "darken the scene", etc.
                To apply a no-op, use "do not change the image". Default is "do not change the image".
            seed: Random seed for reproducible perturbations. Default is 1.
            num_inference_steps: Number of denoising steps. Default is 50.
            text_guidance_scale: Guidance scale for text prompt. Default is 8.0.
            image_guidance_scale: Guidance scale for image conditioning. Default is 2.0.
            device: Device for computation, e.g., "cpu" or "cuda". If None, selects
                CUDA if available, otherwise CPU. Default is None.
        """
        if not self.is_usable():
            raise DiffusionImportError

        super().__init__()

        self.model_name = model_name
        self.prompt = prompt
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.text_guidance_scale = text_guidance_scale
        self.image_guidance_scale = image_guidance_scale
        self.device = device
        self._pipeline: StableDiffusionInstructPix2PixPipeline | None = None

    def _get_device(self) -> str:
        """Get the device to use based on user preference or CUDA availability."""
        if self.device:
            if self.device == "cuda" and not torch.cuda.is_available():
                warnings.warn(
                    "CUDA is not available, but was requested. Falling back to CPU.",
                    UserWarning,
                    stacklevel=2,
                )
                return "cpu"
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _warn_on_cpu_fallback(self, device: str) -> None:
        """Warn user about CPU usage if CUDA is available or if CPU is fallback."""
        if device == "cpu":
            if self.device == "cpu" and torch.cuda.is_available():
                warnings.warn(
                    "Device is set to 'cpu' but CUDA is available. This will be significantly slower.",
                    UserWarning,
                    stacklevel=2,
                )
            elif self.device is None and not torch.cuda.is_available():
                warnings.warn(
                    "CUDA not available, using CPU. This will be significantly slower.",
                    UserWarning,
                    stacklevel=2,
                )

    def _get_pipeline(self) -> StableDiffusionInstructPix2PixPipeline:
        """Get or initialize the diffusion pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        device = self._get_device()

        try:
            # First attempt: standard load, disable low-CPU path to reduce kwargs forwarding
            self._pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                self.model_name,
                safety_checker=None,
                low_cpu_mem_usage=False,
                torch_dtype=torch.float32,
            )
        except TypeError as te:
            # transformers>=4.55 can inject `offload_state_dict` into CLIPTextModel.__init__
            # via a weights_only=True default in the sub-loader. If so, take control of the
            # text encoder load and set weights_only=False explicitly.
            if "offload_state_dict" not in str(te):
                # Unknown TypeError path; bubble it up wrapped.
                raise RuntimeError(
                    f"Failed to load diffusion model '{self.model_name}': {te}",
                ) from te

            text_encoder = self._get_text_encoder()

            self._pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                self.model_name,
                safety_checker=None,
                low_cpu_mem_usage=False,
                text_encoder=text_encoder,  # let diffusers load tokenizer itself
                torch_dtype=torch.float32,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load diffusion model '{self.model_name}': {e}",
            ) from e

        self._finalize_pipeline(device)

        return self._pipeline

    def _get_text_encoder(self) -> CLIPTextModel:
        # Most SD repos store the text encoder in a `text_encoder/` subfolder.
        try:
            text_encoder = CLIPTextModel.from_pretrained(
                self.model_name,
                subfolder="text_encoder",
                low_cpu_mem_usage=False,
                weights_only=False,
            )
        except (OSError, ValueError, RuntimeError):
            # Fallback: load without subfolder if layout differs.
            text_encoder = CLIPTextModel.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=False,
                weights_only=False,
            )

        return text_encoder

    def _finalize_pipeline(self, device: str) -> None:
        self._pipeline = cast(StableDiffusionInstructPix2PixPipeline, self._pipeline)
        self._pipeline = self._pipeline.to(device)

        # keep most of the pipeline in fp16, but run the CLIP image encoder in fp32
        if hasattr(self._pipeline, "image_encoder") and self._pipeline.image_encoder is not None:
            self._pipeline.image_encoder = self._pipeline.image_encoder.to(
                device=self._get_device(),
                dtype=torch.float32,
            )

        self._warn_on_cpu_fallback(device)
        self._pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self._pipeline.scheduler.config,
        )

    def _resize_image(self, image: Image) -> Image:
        """Resize image for Stable Diffusion with proper dimensions.

        Args:
            image: PIL Image to resize

        Returns:
            Resized PIL Image with dimensions suitable for diffusion model
        """
        original_w, original_h = image.size
        # The model was trained on 256x256 images, so it's best suited for images of that size.
        # Documentation states any width over 768px will lead to artifacts.
        min_dimension = 256

        # scale image down to minimum dimension
        scale = min_dimension / min(original_w, original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        # Round to nearest multiple of 8 (required by diffusion model)
        new_w = round(new_w / 8) * 8
        new_h = round(new_h / 8) * 8

        # Lanczos resampling improves image quality: https://mazzo.li/posts/lanczos.html
        # Lancoz is more computationally expensive
        return image.resize((new_w, new_h), Resampling.LANCZOS)

    def _set_seed(self) -> torch.Generator | None:
        """Set random seed for reproducible results."""
        device = self._get_device()
        return torch.Generator(device=device).manual_seed(self.seed)

    @override
    def perturb(
        self,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        **additional_params: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Generate a prompt-guided perturbed image using diffusion models.

        If the prompt is "do not change the image", this method will perform a no-op
        and return the original image and bounding boxes.

        Args:
            image: Input image as a numpy array. PIL will handle format validation.
                Common supported formats: (H, W) grayscale, (H, W, 3) RGB, (H, W, 4) RGBA.
                Input is automatically converted to RGB for processing.
            boxes: Optional iterable of tuples containing AxisAlignedBoundingBox objects
                and their corresponding detection confidence dictionaries.
            additional_params: Additional perturbation keyword arguments (currently unused).

        Returns:
            A tuple containing:
            - Perturbed RGB image as uint8 numpy array (H, W, 3) at diffusion model resolution
            - Updated bounding boxes (currently returned unchanged)

        Raises:
            ValueError: If the input image cannot be converted to PIL format.
            RuntimeError: If the diffusion model fails to load or process the image.
        """
        if self.prompt == "do not change the image":
            return image, boxes

        try:
            pil_image = fromarray(image).convert("RGB")

            resized_image = self._resize_image(pil_image)

            pipeline = self._get_pipeline()
            generator = self._set_seed()

            pipeline_result = pipeline(
                self.prompt,
                image=resized_image,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.text_guidance_scale,
                image_guidance_scale=self.image_guidance_scale,
                generator=generator,
                return_dict=False,
            )
            # pull single image from list of images
            _images, _ = pipeline_result
            generated_image: np.ndarray

            if isinstance(_images, Image):
                generated_image = np.array(_images, dtype=np.uint8)
            else:
                generated_image = np.array(_images[0], dtype=np.uint8)

            return generated_image, boxes

        except Exception as e:
            raise RuntimeError(f"Failed to generate perturbed image: {e}") from e

    @override
    def get_config(self) -> dict[str, Any]:
        """Get the current configuration of the DiffusionPerturber.

        Returns:
            Dictionary containing the current configuration parameters including
            model_name, prompt, seed, and other parameters. The device field shows
            the currently selected device based on CUDA availability.
        """
        return {
            "model_name": self.model_name,
            "prompt": self.prompt,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "text_guidance_scale": self.text_guidance_scale,
            "image_guidance_scale": self.image_guidance_scale,
            "device": self.device,
        }

    @classmethod
    def is_usable(cls) -> bool:
        """Checks if the necessary dependencies (torch, diffusers, and PIL) are available.

        Returns:
            True if torch, diffusers, and PIL are all available; False otherwise.
        """
        return torch_available and diffusion_available and pillow_available
