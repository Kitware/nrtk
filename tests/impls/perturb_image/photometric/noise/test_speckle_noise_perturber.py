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

from nrtk.impls.perturb_image.photometric.noise import SpeckleNoisePerturber
from tests.impls import INPUT_VISDRONE_IMG_FILE_PATH as INPUT_IMG_FILE_PATH
from tests.impls.perturb_image.perturber_tests_mixin import PerturberTestsMixin
from tests.impls.perturb_image.photometric.noise.noise_perturber_test_utils import rng_assertions
from tests.impls.perturb_image.test_perturber_utils import perturber_assertions

test_rng = np.random.default_rng()


@pytest.mark.skimage
class TestSpeckleNoisePerturber(PerturberTestsMixin):
    impl_class = SpeckleNoisePerturber

    def test_consistency(self, psnr_tiff_snapshot: SnapshotAssertion) -> None:
        """Run on a real image to ensure output matches precomputed results."""
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        rng = 42
        mean = 0
        var = 0.05

        # Test callable
        out_img = perturber_assertions(
            perturb=SpeckleNoisePerturber(mean=mean, var=var, rng=rng),
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
                perturb=SpeckleNoisePerturber(mean=0, var=0),
                image=image,
                expected=image,
            )

    def test_default_rng_reproducibility(self) -> None:
        """Ensure results are reproducible with default rng (no rng parameter provided)."""
        dummy_image = test_rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)

        # Test without providing rng (uses default=1)
        inst_1 = SpeckleNoisePerturber()
        out_1, _ = inst_1(image=dummy_image)
        inst_2 = SpeckleNoisePerturber()
        out_2, _ = inst_2(image=dummy_image)
        assert np.array_equal(out_1, out_2)

    @pytest.mark.parametrize("rng", [2])
    def test_rng(self, rng: int) -> None:
        """Ensure results are reproducible when explicit rng is provided."""
        rng_assertions(perturber=SpeckleNoisePerturber, rng=rng)

    @pytest.mark.parametrize(
        ("rng", "mean", "var", "clip"),
        [(42, 0.8, 0.25, True), (np.random.default_rng(12345), 0.3, 0.2, False)],
    )
    def test_configuration(
        self,
        rng: np.random.Generator | int,
        mean: float,
        var: float,
        clip: bool,
    ) -> None:
        """Test configuration stability."""
        inst = SpeckleNoisePerturber(rng=rng, mean=mean, var=var, clip=clip)
        for i in configuration_test_helper(inst):
            assert i.rng == rng
            assert i.mean == mean
            assert i.var == var
            assert i.clip == clip

    @pytest.mark.parametrize(
        ("kwargs", "expectation"),
        [
            ({"var": 0.123, "clip": True}, does_not_raise()),
            ({"var": 0, "clip": False}, does_not_raise()),
            (
                {"var": -10, "clip": True},
                pytest.raises(ValueError, match=r"SpeckleNoisePerturber invalid var"),
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
            SpeckleNoisePerturber(**kwargs)

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
        inst = SpeckleNoisePerturber(rng=42, mean=0.3, var=0.5)
        _, out_boxes = inst.perturb(image=np.ones((256, 256, 3)), boxes=boxes)
        assert boxes == out_boxes
