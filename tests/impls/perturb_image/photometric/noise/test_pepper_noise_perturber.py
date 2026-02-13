from collections.abc import Hashable, Iterable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any

import numpy as np
import pytest
from PIL import Image
from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from syrupy.assertion import SnapshotAssertion

from nrtk.impls.perturb_image.photometric.noise import PepperNoisePerturber
from tests.impls import INPUT_VISDRONE_IMG_FILE_PATH as INPUT_IMG_FILE_PATH
from tests.impls.perturb_image.perturber_tests_mixin import PerturberTestsMixin
from tests.impls.perturb_image.perturber_utils import perturber_assertions
from tests.impls.perturb_image.photometric.noise.noise_perturber_test_utils import seed_assertions

test_rng = np.random.default_rng()


@pytest.mark.skimage
class TestPepperNoisePerturber(PerturberTestsMixin):
    impl_class = PepperNoisePerturber

    def test_consistency(self, psnr_tiff_snapshot: SnapshotAssertion) -> None:
        """Run on a real image to ensure output matches precomputed results."""
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        amount = 0.5

        # Test callable
        out_img = perturber_assertions(
            perturb=PepperNoisePerturber(amount=amount, seed=42, is_static=True),
            image=image,
        )
        psnr_tiff_snapshot.assert_match(out_img)

    @pytest.mark.parametrize(
        ("image", "expectation"),
        [
            (np.zeros((256, 256, 3), dtype=np.uint8), does_not_raise()),
            (np.ones((256, 256, 3), dtype=np.uint8) * 255, does_not_raise()),
            (np.zeros((256, 256, 3), dtype=np.float32), does_not_raise()),
            (np.ones((256, 256, 3), dtype=np.float32), does_not_raise()),
            (np.ones((256, 256, 3), dtype=np.half), does_not_raise()),
            (
                np.ones((256, 356, 3), dtype=np.csingle),
                pytest.raises(
                    NotImplementedError,
                    match=r"Perturb not implemented for",
                ),
            ),
        ],
    )
    def test_no_perturbation(
        self,
        image: np.ndarray,
        expectation: AbstractContextManager,
    ) -> None:
        """Run a dummy image through the perturber with settings for no perturbations, expect to get same image back.

        This attempts to isolate perturber implementation code from external calls to the extent
        that is possible (quantization errors also possible).
        """
        with expectation:
            perturber_assertions(
                perturb=PepperNoisePerturber(amount=0),
                image=image,
                expected=image,
            )

    def test_non_deterministic_default(self) -> None:
        """Verify different results when seed=None (default)."""
        dummy_image = test_rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)

        inst1 = PepperNoisePerturber()
        inst2 = PepperNoisePerturber()
        out1, _ = inst1(image=dummy_image)
        out2, _ = inst2(image=dummy_image)
        assert not np.array_equal(out1, out2)

    @pytest.mark.parametrize("seed", [2])
    def test_seed_reproducibility(self, seed: int) -> None:
        """Ensure results are reproducible when explicit seed is provided."""
        seed_assertions(perturber=PepperNoisePerturber, seed=seed)

    def test_is_static(self) -> None:
        """Verify is_static resets RNG each call."""
        dummy_image = test_rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)
        inst = PepperNoisePerturber(seed=42, is_static=True)
        out1, _ = inst(image=dummy_image)
        out2, _ = inst(image=dummy_image)
        assert np.array_equal(out1, out2)

    def test_is_static_warning(self) -> None:
        """Verify warning when is_static=True with seed=None."""
        with pytest.warns(UserWarning, match="is_static=True has no effect when seed=None"):
            PepperNoisePerturber(seed=None, is_static=True)

    @pytest.mark.parametrize(
        ("seed", "is_static", "amount", "clip"),
        [(42, False, 0.8, True), (None, False, 0.3, False)],
    )
    def test_configuration(
        self,
        seed: int | None,
        is_static: bool,
        amount: float,
        clip: bool,
    ) -> None:
        """Test configuration stability."""
        inst = PepperNoisePerturber(seed=seed, is_static=is_static, amount=amount, clip=clip)
        for i in configuration_test_helper(inst):
            assert i.seed == seed
            assert i.is_static == is_static
            assert i.amount == amount
            assert i.clip == clip

    @pytest.mark.parametrize(
        ("kwargs", "expectation"),
        [
            ({"amount": 0.25, "clip": True}, does_not_raise()),
            ({"amount": 0, "clip": True}, does_not_raise()),
            ({"amount": 1, "clip": False}, does_not_raise()),
            (
                {"amount": 2.5, "clip": True},
                pytest.raises(ValueError, match=r"PepperNoisePerturber invalid amount"),
            ),
            (
                {"amount": -4.2, "clip": False},
                pytest.raises(ValueError, match=r"PepperNoisePerturber invalid amount"),
            ),
        ],
    )
    def test_configuration_bounds(
        self,
        kwargs: dict[str, Any],
        expectation: AbstractContextManager,
    ) -> None:
        """Test that an exception is properly raised (or not) based on argument value."""
        with expectation:
            PepperNoisePerturber(**kwargs)

    @pytest.mark.parametrize(
        "boxes",
        [
            None,
            [(AxisAlignedBoundingBox(min_vertex=(0, 0), max_vertex=(1, 1)), {"test": 0.0})],
            [
                (AxisAlignedBoundingBox(min_vertex=(0, 0), max_vertex=(1, 1)), {"test": 0.0}),
                (AxisAlignedBoundingBox(min_vertex=(2, 2), max_vertex=(3, 3)), {"test2": 1.0}),
            ],
        ],
    )
    def test_perturb_with_boxes(self, boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]) -> None:
        """Test that bounding boxes do not change during perturb."""
        inst = PepperNoisePerturber(seed=42, amount=0.3)
        _, out_boxes = inst.perturb(image=np.ones((256, 256, 3)), boxes=boxes)
        assert boxes == out_boxes
