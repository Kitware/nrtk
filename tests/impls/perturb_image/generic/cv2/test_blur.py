import unittest.mock as mock
from collections.abc import Hashable, Iterable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.impls.perturb_image.generic.cv2.blur import (
    AverageBlurPerturber,
    GaussianBlurPerturber,
    MedianBlurPerturber,
)
from tests.impls.perturb_image.test_perturber_utils import perturber_assertions

rng = np.random.default_rng()


@pytest.mark.skipif(
    not AverageBlurPerturber.is_usable(),
    reason="OpenCV not found. Please install 'nrtk[graphics]' or `nrtk[headless]`.",
)
class TestAverageBlurPerturber:
    def test_consistency(self) -> None:
        """Run on a dummy image to ensure output matches precomputed results."""
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        ksize = 3

        # Test perturb interface directly
        inst = AverageBlurPerturber(ksize=ksize)
        perturber_assertions(perturb=inst.perturb, image=image, expected=EXPECTED_AVERAGE)

        # Test callable
        perturber_assertions(
            perturb=AverageBlurPerturber(ksize=ksize),
            image=image,
            expected=EXPECTED_AVERAGE,
        )

    @pytest.mark.parametrize(
        ("image", "ksize"),
        [
            (rng.integers(0, 255, (256, 256, 3), dtype=np.uint8), 1),
            (np.ones((256, 256, 3), dtype=np.float32), 3),
            (np.ones((256, 256, 3), dtype=np.float64), 5),
        ],
    )
    def test_reproducibility(self, image: np.ndarray, ksize: int) -> None:
        """Ensure results are reproducible."""
        # Test perturb interface directly
        inst = AverageBlurPerturber(ksize=ksize)
        out_image = perturber_assertions(perturb=inst.perturb, image=image, expected=None)
        perturber_assertions(perturb=inst.perturb, image=image, expected=out_image)

        # Test callable
        perturber_assertions(perturb=inst, image=image, expected=out_image)

    @pytest.mark.parametrize("ksize", [4, 6])
    def test_configuration(self, ksize: int) -> None:
        """Test configuration stability."""
        inst = AverageBlurPerturber(ksize=ksize)
        for i in configuration_test_helper(inst):
            assert i.ksize == ksize

    @pytest.mark.parametrize(
        ("kwargs", "expectation"),
        [
            ({"ksize": 2}, does_not_raise()),
            ({"ksize": 1}, does_not_raise()),
            (
                {"ksize": 0},
                pytest.raises(ValueError, match=r"AverageBlurPerturber invalid ksize"),
            ),
        ],
    )
    def test_configuration_bounds(self, kwargs: dict[str, Any], expectation: AbstractContextManager) -> None:
        """Test that an exception is properly raised (or not) based on argument value."""
        with expectation:
            AverageBlurPerturber(**kwargs)

    @pytest.mark.parametrize(
        ("image", "expectation"),
        [
            (np.ones((256, 256)), does_not_raise()),
            (np.ones((256, 256, 3)), does_not_raise()),
            (np.ones((256, 256, 4)), does_not_raise()),
            (
                np.ones((3, 256, 256)),
                pytest.raises(ValueError, match=r"Image is not in expected format"),
            ),
        ],
    )
    def test_perturb_bounds(self, image: np.ndarray, expectation: AbstractContextManager) -> None:
        """Test that an exception is properly raised (or not) based on argument value."""
        inst = AverageBlurPerturber()
        with expectation:
            inst.perturb(image)

    @pytest.mark.parametrize(
        ("boxes"),
        [
            None,
            [(AxisAlignedBoundingBox((0, 0), (1, 1)), {"test": 0.0})],
            [
                (AxisAlignedBoundingBox((0, 0), (1, 1)), {"test": 0.0}),
                (AxisAlignedBoundingBox((2, 2), (3, 3)), {"test2": 1.0}),
            ],
        ],
    )
    def test_perturb_with_boxes(self, boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]) -> None:
        """Test that bounding boxes do not change during perturb."""
        inst = AverageBlurPerturber()
        _, out_boxes = inst.perturb(np.ones((256, 256, 3)), boxes=boxes)
        assert boxes == out_boxes


@pytest.mark.skipif(
    not GaussianBlurPerturber.is_usable(),
    reason="OpenCV not found. Please install 'nrtk[graphics]' or `nrtk[headless]`.",
)
class TestGaussianBlurPerturber:
    def test_consistency(self) -> None:
        """Run on a dummy image to ensure output matches precomputed results."""
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        ksize = 3

        # Test perturb interface directly
        inst = GaussianBlurPerturber(ksize=ksize)
        perturber_assertions(perturb=inst.perturb, image=image, expected=EXPECTED_GAUSSIAN)

        # Test callable
        perturber_assertions(
            perturb=GaussianBlurPerturber(ksize=ksize),
            image=image,
            expected=EXPECTED_GAUSSIAN,
        )

    @pytest.mark.parametrize(
        ("image", "ksize"),
        [
            (rng.integers(0, 255, (256, 256, 3), dtype=np.uint8), 1),
            (np.ones((256, 256, 3), dtype=np.float32), 3),
            (np.ones((256, 256, 3), dtype=np.float64), 5),
        ],
    )
    def test_reproducibility(self, image: np.ndarray, ksize: int) -> None:
        """Ensure results are reproducible."""
        # Test perturb interface directly
        inst = GaussianBlurPerturber(ksize=ksize)
        out_image = perturber_assertions(perturb=inst.perturb, image=image, expected=None)
        perturber_assertions(perturb=inst.perturb, image=image, expected=out_image)

        # Test callable
        perturber_assertions(perturb=inst, image=image, expected=out_image)

    @pytest.mark.parametrize("ksize", [5, 7])
    def test_configuration(self, ksize: int) -> None:
        """Test configuration stability."""
        inst = GaussianBlurPerturber(ksize=ksize)
        for i in configuration_test_helper(inst):
            assert i.ksize == ksize

    @pytest.mark.parametrize(
        ("kwargs", "expectation"),
        [
            (
                {"ksize": 4},
                pytest.raises(ValueError, match=r"GaussianBlurPerturber invalid ksize"),
            ),
            ({"ksize": 3}, does_not_raise()),
            ({"ksize": 1}, does_not_raise()),
            (
                {"ksize": 0},
                pytest.raises(ValueError, match=r"GaussianBlurPerturber invalid ksize"),
            ),
            (
                {"ksize": -1},
                pytest.raises(ValueError, match=r"GaussianBlurPerturber invalid ksize"),
            ),
        ],
    )
    def test_configuration_bounds(self, kwargs: dict[str, Any], expectation: AbstractContextManager) -> None:
        """Test that bounding boxes do not change during perturb."""
        with expectation:
            GaussianBlurPerturber(**kwargs)

    @pytest.mark.parametrize(
        ("image", "expectation"),
        [
            (np.ones((256, 256)), does_not_raise()),
            (np.ones((256, 256, 3)), does_not_raise()),
            (np.ones((256, 256, 4)), does_not_raise()),
            (
                np.ones((3, 256, 256)),
                pytest.raises(ValueError, match=r"Image is not in expected format"),
            ),
        ],
    )
    def test_perturb_bounds(self, image: np.ndarray, expectation: AbstractContextManager) -> None:
        """Test that an exception is properly raised (or not) based on argument value."""
        inst = GaussianBlurPerturber()
        with expectation:
            inst.perturb(image)

    @pytest.mark.parametrize(
        "boxes",
        [
            None,
            [(AxisAlignedBoundingBox((0, 0), (1, 1)), {"test": 0.0})],
            [
                (AxisAlignedBoundingBox((0, 0), (1, 1)), {"test": 0.0}),
                (AxisAlignedBoundingBox((2, 2), (3, 3)), {"test2": 1.0}),
            ],
        ],
    )
    def test_perturb_with_boxes(self, boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]) -> None:
        """Test that bounding boxes do not change during perturb."""
        inst = GaussianBlurPerturber()
        _, out_boxes = inst.perturb(np.ones((256, 256, 3)), boxes=boxes)
        assert boxes == out_boxes


@pytest.mark.skipif(
    not MedianBlurPerturber.is_usable(),
    reason="OpenCV not found. Please install 'nrtk[graphics]' or `nrtk[headless]`.",
)
class TestMedianBlurPerturber:
    def test_consistency(self) -> None:
        """Run on a dummy image to ensure output matches precomputed results."""
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        ksize = 3

        # Test perturb interface directly
        inst = MedianBlurPerturber(ksize=ksize)
        perturber_assertions(perturb=inst.perturb, image=image, expected=EXPECTED_MEDIAN)

        # Test callable
        perturber_assertions(
            perturb=MedianBlurPerturber(ksize=ksize),
            image=image,
            expected=EXPECTED_MEDIAN,
        )

    @pytest.mark.parametrize(
        ("image", "ksize"),
        [
            (rng.integers(0, 255, (256, 256, 3), dtype=np.uint8), 3),
            (np.ones((256, 256, 3), dtype=np.float32), 5),
        ],
    )
    def test_reproducibility(self, image: np.ndarray, ksize: int) -> None:
        """Ensure results are reproducible."""
        # Test perturb interface directly
        inst = MedianBlurPerturber(ksize=ksize)
        out_image = perturber_assertions(perturb=inst.perturb, image=image, expected=None)
        perturber_assertions(perturb=inst.perturb, image=image, expected=out_image)

        # Test callable
        perturber_assertions(perturb=inst, image=image, expected=out_image)

    @pytest.mark.parametrize("ksize", [5, 7])
    def test_configuration(self, ksize: int) -> None:
        """Test configuration stability."""
        inst = MedianBlurPerturber(ksize=ksize)
        for i in configuration_test_helper(inst):
            assert i.ksize == ksize

    @pytest.mark.parametrize(
        ("kwargs", "expectation"),
        [
            ({"ksize": 5}, does_not_raise()),
            (
                {"ksize": 4},
                pytest.raises(ValueError, match=r"MedianBlurPerturber invalid ksize"),
            ),
            ({"ksize": 3}, does_not_raise()),
            (
                {"ksize": 2},
                pytest.raises(ValueError, match=r"MedianBlurPerturber invalid ksize"),
            ),
            (
                {"ksize": 1},
                pytest.raises(ValueError, match=r"MedianBlurPerturber invalid ksize"),
            ),
        ],
    )
    def test_configuration_bounds(self, kwargs: dict[str, Any], expectation: AbstractContextManager) -> None:
        """Test that an exception is properly raised (or not) based on argument value."""
        with expectation:
            MedianBlurPerturber(**kwargs)

    @pytest.mark.parametrize(
        ("image", "expectation"),
        [
            (np.ones((256, 256), dtype=np.float32), does_not_raise()),
            (np.ones((256, 256, 3), dtype=np.float32), does_not_raise()),
            (np.ones((256, 256, 4), dtype=np.float32), does_not_raise()),
            (
                np.ones((3, 256, 256)),
                pytest.raises(ValueError, match=r"Image is not in expected format"),
            ),
        ],
    )
    def test_perturb_bounds(self, image: np.ndarray, expectation: AbstractContextManager) -> None:
        """Test that an exception is properly raised (or not) based on argument value."""
        inst = MedianBlurPerturber(ksize=5)
        with expectation:
            inst.perturb(image)

    @pytest.mark.parametrize(
        ("boxes"),
        [
            None,
            [(AxisAlignedBoundingBox((0, 0), (1, 1)), {"test": 0.0})],
            [
                (AxisAlignedBoundingBox((0, 0), (1, 1)), {"test": 0.0}),
                (AxisAlignedBoundingBox((2, 2), (3, 3)), {"test2": 1.0}),
            ],
        ],
    )
    def test_perturb_with_boxes(self, boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]) -> None:
        """Test that bounding boxes do not change during perturb."""
        inst = MedianBlurPerturber(ksize=5)
        _, out_boxes = inst.perturb(np.ones((256, 256, 3), dtype=np.float32), boxes=boxes)
        assert boxes == out_boxes


@mock.patch.object(MedianBlurPerturber, "is_usable")
def test_missing_deps(mock_is_usable: MagicMock) -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    mock_is_usable.return_value = False
    assert not MedianBlurPerturber.is_usable()
    with pytest.raises(ImportError, match=r"OpenCV not found"):
        MedianBlurPerturber()


EXPECTED_AVERAGE = np.array([[4, 4, 4], [5, 5, 5], [6, 6, 6]], dtype=np.uint8)
EXPECTED_GAUSSIAN = np.array([[3, 4, 4], [5, 5, 6], [6, 7, 7]], dtype=np.uint8)
EXPECTED_MEDIAN = np.array([[2, 3, 3], [4, 5, 6], [7, 7, 8]], dtype=np.uint8)
