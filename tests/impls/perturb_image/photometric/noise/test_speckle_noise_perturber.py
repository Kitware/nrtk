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
from tests.impls.perturb_image.perturber_utils import perturber_assertions
from tests.impls.perturb_image.photometric.noise.noise_perturber_test_utils import seed_assertions

test_rng = np.random.default_rng()


@pytest.mark.skimage
class TestSpeckleNoisePerturber(PerturberTestsMixin):
    impl_class = SpeckleNoisePerturber

    def test_consistency(self, psnr_tiff_snapshot: SnapshotAssertion) -> None:
        """Run on a real image to ensure output matches precomputed results."""
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        mean = 0
        var = 0.05

        # Test callable
        out_img = perturber_assertions(
            perturb=SpeckleNoisePerturber(mean=mean, var=var, seed=42, is_static=True),
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

    def test_non_deterministic_default(self) -> None:
        """Verify different results when seed=None (default)."""
        dummy_image = test_rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)

        inst1 = SpeckleNoisePerturber()
        inst2 = SpeckleNoisePerturber()
        out1, _ = inst1(image=dummy_image)
        out2, _ = inst2(image=dummy_image)
        assert not np.array_equal(out1, out2)

    @pytest.mark.parametrize("seed", [2])
    def test_seed_reproducibility(self, seed: int) -> None:
        """Ensure results are reproducible when explicit seed is provided."""
        seed_assertions(perturber=SpeckleNoisePerturber, seed=seed)

    def test_is_static(self) -> None:
        """Verify is_static resets RNG each call."""
        dummy_image = test_rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)
        inst = SpeckleNoisePerturber(seed=42, is_static=True)
        out1, _ = inst(image=dummy_image)
        out2, _ = inst(image=dummy_image)
        assert np.array_equal(out1, out2)

    def test_is_static_warning(self) -> None:
        """Verify warning when is_static=True with seed=None."""
        with pytest.warns(UserWarning, match="is_static=True has no effect when seed=None"):
            SpeckleNoisePerturber(seed=None, is_static=True)

    @pytest.mark.parametrize(
        ("seed", "is_static", "mean", "var", "clip"),
        [(42, False, 0.8, 0.25, True), (None, False, 0.3, 0.2, False)],
    )
    def test_configuration(
        self,
        seed: int | None,
        is_static: bool,
        mean: float,
        var: float,
        clip: bool,
    ) -> None:
        """Test configuration stability."""
        inst = SpeckleNoisePerturber(seed=seed, is_static=is_static, mean=mean, var=var, clip=clip)
        for i in configuration_test_helper(inst):
            assert i.seed == seed
            assert i.is_static == is_static
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
        inst = SpeckleNoisePerturber(seed=42, mean=0.3, var=0.5)
        _, out_boxes = inst.perturb(image=np.ones((256, 256, 3)), boxes=boxes)
        assert boxes == out_boxes
