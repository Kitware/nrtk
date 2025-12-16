"""Unit tests for DiffusionPerturber implementation.

This module contains unit tests for the DiffusionPerturber class,
ensuring proper interface compliance, configuration handling, and expected behaviors.
"""

from __future__ import annotations

import socket
import warnings
from collections.abc import Hashable, Iterable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image
from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.impls.perturb.generative.diffusion_perturber import DiffusionPerturber
from nrtk.utils._exceptions import DiffusionImportError
from tests.impls.perturb.test_perturber_utils import perturber_assertions


def internet_available() -> bool:
    """Check if internet is available by trying to connect to a known host."""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=1)
        return True
    except (TimeoutError, OSError):
        return False


def cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
    except ImportError:
        return False

    return torch.cuda.is_available()


@pytest.mark.skipif(not DiffusionPerturber.is_usable(), reason=str(DiffusionImportError()))
class TestDiffusionPerturber:
    """Test class for DiffusionPerturber functionality."""

    @pytest.mark.parametrize(
        ("boxes"),
        [
            None,
            [(AxisAlignedBoundingBox(min_vertex=(0, 0), max_vertex=(1, 1)), {"test": 0.0})],
            [
                (AxisAlignedBoundingBox(min_vertex=(0, 0), max_vertex=(10, 10)), {"class1": 0.8}),
                (AxisAlignedBoundingBox(min_vertex=(50, 50), max_vertex=(100, 100)), {"class2": 0.9}),
            ],
        ],
    )
    @pytest.mark.parametrize("device", ["cuda", "cpu", None])
    @patch("nrtk.impls.perturb.generative.diffusion_perturber.torch")
    @patch("nrtk.impls.perturb.generative.diffusion_perturber.StableDiffusionInstructPix2PixPipeline")
    @patch("nrtk.impls.perturb.generative.diffusion_perturber.EulerAncestralDiscreteScheduler")
    def test_perturb_with_boxes(
        self,
        mock_scheduler_class: MagicMock,
        mock_pipeline_class: MagicMock,
        mock_torch: MagicMock,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None,
        device: str | None,
    ) -> None:
        """Test that bounding boxes are returned unchanged and mocks are called correctly."""
        mock_pipeline = MagicMock()
        mock_image = np.zeros((256, 256, 3), dtype=np.uint8)
        mock_pipeline.return_value = ([mock_image], False)
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline

        mock_pipeline.scheduler.config = {"test": "config"}
        mock_scheduler_class.from_config.return_value = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        # Prompt is set to test to prevent a no-op
        perturber = DiffusionPerturber(model_name="test/model", device=device, prompt="test")
        image = np.ones((256, 256, 3), dtype=np.uint8)

        perturbed_image, output_boxes = perturber.perturb(image=image, boxes=boxes)

        expected_device = perturber._get_device()

        # robust to extra kwargs; still checks the invariants we care about
        mock_pipeline_class.from_pretrained.assert_called_once()
        fp_args, fp_kwargs = mock_pipeline_class.from_pretrained.call_args
        assert fp_args[0] == "test/model"
        assert fp_kwargs["safety_checker"] is None

        mock_pipeline.to.assert_called_once_with(expected_device)
        mock_scheduler_class.from_config.assert_called_once_with({"test": "config"})

        # Check pipeline call arguments
        call_args, call_kwargs = mock_pipeline.call_args
        assert call_args[0] == perturber.prompt
        assert isinstance(call_kwargs["image"], Image.Image)
        assert call_kwargs["image"].size == (256, 256)
        assert call_kwargs["num_inference_steps"] == perturber.num_inference_steps
        assert call_kwargs["guidance_scale"] == perturber.text_guidance_scale
        assert call_kwargs["image_guidance_scale"] == perturber.image_guidance_scale
        mock_torch.Generator.assert_called_once_with(device=expected_device)

        assert perturbed_image.shape == (256, 256, 3)
        assert output_boxes == boxes

    @patch("nrtk.impls.perturb.generative.diffusion_perturber.torch")
    @patch("nrtk.impls.perturb.generative.diffusion_perturber.StableDiffusionInstructPix2PixPipeline")
    @patch("nrtk.impls.perturb.generative.diffusion_perturber.EulerAncestralDiscreteScheduler")
    def test_default_seed_reproducibility(
        self,
        mock_scheduler_class: MagicMock,
        mock_pipeline_class: MagicMock,
        mock_torch: MagicMock,
    ) -> None:
        """Ensure results are reproducible with default seed (no seed parameter provided)."""
        image = np.random.default_rng(1).integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)

        # Setup mocks
        mock_pipeline = MagicMock()
        mock_result_image = np.random.default_rng(1).integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)
        mock_pipeline.return_value = ([mock_result_image], False)
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline
        mock_pipeline.scheduler.config = {}
        mock_scheduler_class.from_config.return_value = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        # Test perturb interface directly without providing seed (uses default=1)
        inst = DiffusionPerturber(model_name="test/model", prompt="test")
        out_image = perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=None,
        )

        # Create new instance without seed
        inst = DiffusionPerturber(model_name="test/model", prompt="test")
        perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=out_image,
        )

        # Test callable
        inst = DiffusionPerturber(model_name="test/model", prompt="test")
        perturber_assertions(
            perturb=inst,
            image=image,
            expected=out_image,
        )

    @pytest.mark.parametrize(
        ("image", "seed"),
        [
            (np.random.default_rng(2).integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8), 2),
        ],
    )
    @patch("nrtk.impls.perturb.generative.diffusion_perturber.torch")
    @patch("nrtk.impls.perturb.generative.diffusion_perturber.StableDiffusionInstructPix2PixPipeline")
    @patch("nrtk.impls.perturb.generative.diffusion_perturber.EulerAncestralDiscreteScheduler")
    def test_reproducibility(
        self,
        mock_scheduler_class: MagicMock,
        mock_pipeline_class: MagicMock,
        mock_torch: MagicMock,
        image: np.ndarray,
        seed: int,
    ) -> None:
        """Ensure results are reproducible when explicit seed is provided."""
        # Setup mocks
        mock_pipeline = MagicMock()
        mock_result_image = np.random.default_rng(seed).integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)
        mock_pipeline.return_value = ([mock_result_image], False)
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline
        mock_pipeline.scheduler.config = {}
        mock_scheduler_class.from_config.return_value = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        # Test perturb interface directly
        inst = DiffusionPerturber(model_name="test/model", prompt="test", seed=seed)
        out_image = perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=None,
        )

        # Create new instance with same seed
        inst = DiffusionPerturber(model_name="test/model", prompt="test", seed=seed)
        perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=out_image,
        )

        # Test callable
        inst = DiffusionPerturber(model_name="test/model", prompt="test", seed=seed)
        perturber_assertions(
            perturb=inst,
            image=image,
            expected=out_image,
        )

    @pytest.mark.parametrize(
        (
            "model_name",
            "prompt",
            "seed",
            "num_inference_steps",
            "text_guidance_scale",
            "image_guidance_scale",
            "device",
        ),
        [
            ("test/model", "add rain", 42, 50, 8.0, 2.0, "cuda"),
            ("custom/model", "make foggy", 123, 30, 7.5, 1.5, None),
        ],
    )
    def test_configuration(
        self,
        model_name: str,
        prompt: str,
        seed: int,
        num_inference_steps: int,
        text_guidance_scale: float,
        image_guidance_scale: float,
        device: str | None,
    ) -> None:
        """Test configuration stability."""
        inst = DiffusionPerturber(
            model_name=model_name,
            prompt=prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            image_guidance_scale=image_guidance_scale,
            device=device,
        )

        for perturber in configuration_test_helper(inst):
            assert perturber.model_name == model_name
            assert perturber.prompt == prompt
            assert perturber.seed == seed
            assert perturber.num_inference_steps == num_inference_steps
            assert perturber.text_guidance_scale == text_guidance_scale
            assert perturber.image_guidance_scale == image_guidance_scale
            assert perturber.device == device

    @patch("nrtk.impls.perturb.generative.diffusion_perturber.StableDiffusionInstructPix2PixPipeline")
    def test_pipeline_loading_error(self, mock_pipeline_class: MagicMock) -> None:
        """Test that pipeline loading errors are properly handled."""
        mock_pipeline_class.from_pretrained.side_effect = Exception("Model not found")

        perturber = DiffusionPerturber(model_name="nonexistent/model")

        with pytest.raises(RuntimeError, match="Failed to load diffusion model"):
            perturber._get_pipeline()

    @pytest.mark.parametrize(
        ("device_requested", "cuda_available", "expected_device", "warning_expected"),
        [
            ("cuda", True, "cuda", False),
            ("cuda", False, "cpu", True),
            ("cpu", True, "cpu", True),
            ("cpu", False, "cpu", False),
            (None, True, "cuda", False),
            (None, False, "cpu", True),
        ],
    )
    @patch("nrtk.impls.perturb.generative.diffusion_perturber.StableDiffusionInstructPix2PixPipeline")
    @patch("nrtk.impls.perturb.generative.diffusion_perturber.EulerAncestralDiscreteScheduler")
    @patch("nrtk.impls.perturb.generative.diffusion_perturber.torch")
    def test_device_selection_and_warnings(
        self,
        mock_torch: MagicMock,
        mock_scheduler_class: MagicMock,
        mock_pipeline_class: MagicMock,
        device_requested: str | None,
        cuda_available: bool,
        expected_device: str,
        warning_expected: bool,
    ) -> None:
        """Test device selection logic and associated warnings."""
        mock_torch.cuda.is_available.return_value = cuda_available

        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline
        mock_pipeline.scheduler.config = {}
        mock_scheduler_class.from_config.return_value = MagicMock()

        perturber = DiffusionPerturber(device=device_requested)

        if warning_expected:
            with pytest.warns(UserWarning, match=r"(?i)cpu"):
                perturber._get_pipeline()
        else:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", UserWarning)
                perturber._get_pipeline()
                assert not w, f"Expected no UserWarnings, but got: {[str(warn.message) for warn in w]}"

        mock_pipeline.to.assert_called_with(expected_device)

    # @pytest.mark.parametrize(
    #     ("prompt", "image_guidance_scale", "text_guidance_scale", "seed"),
    #     [
    #         ("add rain to the image", 2.0, 8.0, 42),
    #     ],
    # )
    # @pytest.mark.skipif(not internet_available(), reason="Internet connection not available.")
    # @pytest.mark.skipif(not cuda_available(), reason="CUDA not available.")
    # def test_regression(
    #     self,
    #     snapshot: SnapshotAssertion,
    #     prompt: str,
    #     image_guidance_scale: float,
    #     text_guidance_scale: float,
    #     seed: int,
    # ) -> None:
    #     """Test the full perturber pipeline to ensure it runs."""
    #     perturber = DiffusionPerturber(
    #         prompt=prompt,
    #         num_inference_steps=2,  # just a few steps to test the pipeline
    #         image_guidance_scale=image_guidance_scale,
    #         text_guidance_scale=text_guidance_scale,
    #         seed=seed,
    #     )

    #     image = np.full((256, 256, 3), 128, dtype=np.uint8)
    #     perturbed_image = perturber_assertions(perturber.perturb, image)

    #     # make sure the image has been modified
    #     assert not np.array_equal(image, perturbed_image)

    #     assert TIFFImageSnapshotExtension.ndarray2bytes(perturbed_image) == snapshot(
    #         extension_class=TIFFImageSnapshotExtension,
    #     )


@pytest.mark.parametrize(
    "target",
    [
        "nrtk.impls.perturb.generative.diffusion_perturber.torch_available",
        "nrtk.impls.perturb.generative.diffusion_perturber.diffusion_available",
        "nrtk.impls.perturb.generative.diffusion_perturber.pillow_available",
    ],
)
def test_missing_deps(target: str) -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    with patch(target, False):
        assert not DiffusionPerturber.is_usable()
        with pytest.raises(DiffusionImportError):
            DiffusionPerturber()
