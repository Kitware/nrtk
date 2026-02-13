"""Unit tests for DiffusionPerturber implementation.

This module contains unit tests for the DiffusionPerturber class,
ensuring proper interface compliance, configuration handling, and expected behaviors.
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Iterable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image
from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.impls.perturb_image.generative import DiffusionPerturber
from tests.impls.perturb_image.perturber_tests_mixin import PerturberTestsMixin
from tests.impls.perturb_image.perturber_utils import perturber_assertions

_BASE_TORCH = "nrtk.impls.perturb_image._base._torch_random_perturb_image.torch"
_DIFFUSION_TORCH = "nrtk.impls.perturb_image.generative._diffusion_perturber.torch"
_DIFFUSION_PIPELINE = "nrtk.impls.perturb_image.generative._diffusion_perturber.StableDiffusionInstructPix2PixPipeline"
_DIFFUSION_SCHEDULER = "nrtk.impls.perturb_image.generative._diffusion_perturber.EulerAncestralDiscreteScheduler"


@pytest.mark.diffusion
class TestDiffusionPerturber(PerturberTestsMixin):
    """Test class for DiffusionPerturber functionality."""

    impl_class = DiffusionPerturber

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
    @patch(_BASE_TORCH)
    @patch(_DIFFUSION_TORCH)
    @patch(_DIFFUSION_PIPELINE)
    @patch(_DIFFUSION_SCHEDULER)
    def test_perturb_with_boxes(
        self,
        mock_scheduler_class: MagicMock,
        mock_pipeline_class: MagicMock,
        mock_torch: MagicMock,
        mock_base_torch: MagicMock,
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

        # Prompt is set to test to prevent a no-op; seed is set to enable Generator creation
        perturber = DiffusionPerturber(model_name="test/model", device=device, prompt="test", seed=42)
        image = np.ones((256, 256, 3), dtype=np.uint8)

        perturbed_image, output_boxes = perturber.perturb(image=image, boxes=boxes)

        expected_device = perturber._get_device()

        mock_base_torch.Generator.assert_called_with(device=expected_device)

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

        assert perturbed_image.shape == (256, 256, 3)
        assert output_boxes == boxes

    @patch(_BASE_TORCH)
    @patch(_DIFFUSION_TORCH)
    @patch(_DIFFUSION_PIPELINE)
    @patch(_DIFFUSION_SCHEDULER)
    def test_non_deterministic_default(
        self,
        mock_scheduler_class: MagicMock,
        mock_pipeline_class: MagicMock,
        mock_torch: MagicMock,
        mock_base_torch: MagicMock,  # noqa: ARG002
    ) -> None:
        """Verify different results when seed=None (default)."""
        image = np.random.default_rng(1).integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)

        # Setup mocks - return different images for different calls
        mock_pipeline = MagicMock()
        mock_result_image1 = np.random.default_rng(1).integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)
        mock_result_image2 = np.random.default_rng(2).integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)
        mock_pipeline.side_effect = [([mock_result_image1], False), ([mock_result_image2], False)]
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline
        mock_pipeline.scheduler.config = {}
        mock_scheduler_class.from_config.return_value = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        # Create two instances with default seed=None
        inst1 = DiffusionPerturber(model_name="test/model", prompt="test")
        inst2 = DiffusionPerturber(model_name="test/model", prompt="test")
        out1, _ = inst1.perturb(image=image)
        out2, _ = inst2.perturb(image=image)
        # Results should be different with non-deterministic default (mocked)
        assert not np.array_equal(out1, out2)

    @pytest.mark.parametrize(
        ("image", "seed"),
        [
            (np.random.default_rng(2).integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8), 2),
        ],
    )
    @patch(_BASE_TORCH)
    @patch(_DIFFUSION_TORCH)
    @patch(_DIFFUSION_PIPELINE)
    @patch(_DIFFUSION_SCHEDULER)
    def test_seed_reproducibility(
        self,
        mock_scheduler_class: MagicMock,
        mock_pipeline_class: MagicMock,
        mock_torch: MagicMock,
        mock_base_torch: MagicMock,  # noqa: ARG002
        image: np.ndarray,
        seed: int,
    ) -> None:
        """Verify same results with explicit seed."""
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

    @patch(_BASE_TORCH)
    @patch(_DIFFUSION_TORCH)
    @patch(_DIFFUSION_PIPELINE)
    @patch(_DIFFUSION_SCHEDULER)
    def test_is_static(
        self,
        mock_scheduler_class: MagicMock,
        mock_pipeline_class: MagicMock,
        mock_torch: MagicMock,
        mock_base_torch: MagicMock,  # noqa: ARG002
    ) -> None:
        """Verify is_static resets RNG each call."""
        image = np.random.default_rng(1).integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)

        # Setup mocks
        mock_pipeline = MagicMock()
        mock_result_image = np.random.default_rng(42).integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)
        mock_pipeline.return_value = ([mock_result_image], False)
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline
        mock_pipeline.scheduler.config = {}
        mock_scheduler_class.from_config.return_value = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        inst = DiffusionPerturber(model_name="test/model", prompt="test", seed=42, is_static=True)
        out1 = perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=None,
        )
        # Same result on second call with is_static
        perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=out1,
        )

    def test_is_static_warning(self) -> None:
        """Verify warning when is_static=True with seed=None."""
        with pytest.warns(UserWarning, match="is_static=True has no effect"):
            DiffusionPerturber(model_name="test/model", prompt="test", seed=None, is_static=True)

    @pytest.mark.parametrize(
        (
            "model_name",
            "prompt",
            "seed",
            "num_inference_steps",
            "text_guidance_scale",
            "image_guidance_scale",
            "device",
            "is_static",
        ),
        [
            ("test/model", "add rain", 42, 50, 8.0, 2.0, "cuda", False),
            ("custom/model", "make foggy", 123, 30, 7.5, 1.5, None, True),
            ("test/model", "add snow", None, 25, 8.0, 2.0, "cpu", False),
        ],
    )
    @patch(_BASE_TORCH)
    @patch(_DIFFUSION_TORCH)
    def test_configuration(
        self,
        mock_torch: MagicMock,
        mock_base_torch: MagicMock,  # noqa: ARG002
        model_name: str,
        prompt: str,
        seed: int | None,
        num_inference_steps: int,
        text_guidance_scale: float,
        image_guidance_scale: float,
        device: str | None,
        is_static: bool,
    ) -> None:
        """Test configuration stability."""
        mock_torch.cuda.is_available.return_value = True

        inst = DiffusionPerturber(
            model_name=model_name,
            prompt=prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            image_guidance_scale=image_guidance_scale,
            device=device,
            is_static=is_static,
        )

        for perturber in configuration_test_helper(inst):
            assert perturber.model_name == model_name
            assert perturber.prompt == prompt
            assert perturber.seed == seed
            assert perturber.num_inference_steps == num_inference_steps
            assert perturber.text_guidance_scale == text_guidance_scale
            assert perturber.image_guidance_scale == image_guidance_scale
            assert perturber._device_config == device
            assert perturber.is_static == is_static

    @patch(_DIFFUSION_PIPELINE)
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
    @patch(_BASE_TORCH)
    @patch(_DIFFUSION_PIPELINE)
    @patch(_DIFFUSION_SCHEDULER)
    @patch(_DIFFUSION_TORCH)
    def test_device_selection_and_warnings(
        self,
        mock_torch: MagicMock,
        mock_scheduler_class: MagicMock,
        mock_pipeline_class: MagicMock,
        mock_base_torch: MagicMock,  # noqa: ARG002
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
