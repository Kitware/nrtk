import numpy as np
import pytest
from contextlib import nullcontext as does_not_raise
from smqtk_core.configuration import configuration_test_helper
from typing import Any, ContextManager, Dict

from nrtk.impls.perturb_image.generic.cv2.blur import (
    AverageBlurPerturber,
    GaussianBlurPerturber,
    MedianBlurPerturber
)

from ...test_perturber_utils import perturber_assertions


class TestAverageBlurPerturber:
    def test_consistency(self) -> None:
        """
        Run on a dummy image to ensure output matches precomputed results.
        """
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        ksize = 3

        # Test perturb interface directly
        inst = AverageBlurPerturber(ksize=ksize)
        perturber_assertions(perturb=inst.perturb, image=image, expected=EXPECTED_AVERAGE)

        # Test callable
        perturber_assertions(
            perturb=AverageBlurPerturber(ksize=ksize),
            image=image,
            expected=EXPECTED_AVERAGE
        )

    @pytest.mark.parametrize("image, ksize", [
        (np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8), 1),
        (np.ones((256, 256, 3), dtype=np.float32), 3),
        (np.ones((256, 256, 3), dtype=np.float64), 5),
    ])
    def test_reproducibility(self, image: np.ndarray, ksize: int) -> None:
        """
        Ensure results are reproducible.
        """
        # Test perturb interface directly
        inst = AverageBlurPerturber(ksize=ksize)
        out_image = perturber_assertions(perturb=inst.perturb, image=image, expected=None)
        perturber_assertions(perturb=inst.perturb, image=image, expected=out_image)

        # Test callable
        perturber_assertions(perturb=inst, image=image, expected=out_image)

    @pytest.mark.parametrize("ksize", [4, 6])
    def test_configuration(self, ksize: int) -> None:
        """
        Test configuration stability.
        """
        inst = AverageBlurPerturber(ksize=ksize)
        for i in configuration_test_helper(inst):
            assert i.ksize == ksize

    @pytest.mark.parametrize("kwargs, expectation", [
        ({"ksize": 2}, does_not_raise()),
        ({"ksize": 1}, does_not_raise()),
        ({"ksize": 0}, pytest.raises(ValueError, match=r"AverageBlurPerturber invalid ksize"))
    ])
    def test_configuration_bounds(self, kwargs: Dict[str, Any], expectation: ContextManager) -> None:
        """
        Test that an exception is properly raised (or not) based on argument value.
        """
        with expectation:
            AverageBlurPerturber(**kwargs)


class TestGaussianBlurPerturber:
    def test_consistency(self) -> None:
        """
        Run on a dummy image to ensure output matches precomputed results.
        """
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        ksize = 3

        # Test perturb interface directly
        inst = GaussianBlurPerturber(ksize=ksize)
        perturber_assertions(perturb=inst.perturb, image=image, expected=EXPECTED_GAUSSIAN)

        # Test callable
        perturber_assertions(
            perturb=GaussianBlurPerturber(ksize=ksize),
            image=image,
            expected=EXPECTED_GAUSSIAN
        )

    @pytest.mark.parametrize("image, ksize", [
        (np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8), 1),
        (np.ones((256, 256, 3), dtype=np.float32), 3),
        (np.ones((256, 256, 3), dtype=np.float64), 5),
    ])
    def test_reproducibility(self, image: np.ndarray, ksize: int) -> None:
        """
        Ensure results are reproducible.
        """
        # Test perturb interface directly
        inst = GaussianBlurPerturber(ksize=ksize)
        out_image = perturber_assertions(perturb=inst.perturb, image=image, expected=None)
        perturber_assertions(perturb=inst.perturb, image=image, expected=out_image)

        # Test callable
        perturber_assertions(perturb=inst, image=image, expected=out_image)

    @pytest.mark.parametrize("ksize", [5, 7])
    def test_configuration(self, ksize: int) -> None:
        """
        Test configuration stability.
        """
        inst = GaussianBlurPerturber(ksize=ksize)
        for i in configuration_test_helper(inst):
            assert i.ksize == ksize

    @pytest.mark.parametrize("kwargs, expectation", [
        ({"ksize": 4}, pytest.raises(ValueError, match=r"GaussianBlurPerturber invalid ksize")),
        ({"ksize": 3}, does_not_raise()),
        ({"ksize": 1}, does_not_raise()),
        ({"ksize": 0}, pytest.raises(ValueError, match=r"GaussianBlurPerturber invalid ksize")),
        ({"ksize": -1}, pytest.raises(ValueError, match=r"GaussianBlurPerturber invalid ksize")),
    ])
    def test_configuration_bounds(self, kwargs: Dict[str, Any], expectation: ContextManager) -> None:
        """
        Test that an exception is properly raised (or not) based on argument value.
        """
        with expectation:
            GaussianBlurPerturber(**kwargs)


class TestMedianBlurPerturber:
    def test_consistency(self) -> None:
        """
        Run on a dummy image to ensure output matches precomputed results.
        """
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        ksize = 3

        # Test perturb interface directly
        inst = MedianBlurPerturber(ksize=ksize)
        perturber_assertions(perturb=inst.perturb, image=image, expected=EXPECTED_MEDIAN)

        # Test callable
        perturber_assertions(
            perturb=MedianBlurPerturber(ksize=ksize),
            image=image,
            expected=EXPECTED_MEDIAN
        )

    @pytest.mark.parametrize("image, ksize", [
        (np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8), 3),
        (np.ones((256, 256, 3), dtype=np.float32), 5)
    ])
    def test_reproducibility(self, image: np.ndarray, ksize: int) -> None:
        """
        Ensure results are reproducible.
        """
        # Test perturb interface directly
        inst = MedianBlurPerturber(ksize=ksize)
        out_image = perturber_assertions(perturb=inst.perturb, image=image, expected=None)
        perturber_assertions(perturb=inst.perturb, image=image, expected=out_image)

        # Test callable
        perturber_assertions(perturb=inst, image=image, expected=out_image)

    @pytest.mark.parametrize("ksize", [5, 7])
    def test_configuration(self, ksize: int) -> None:
        """
        Test configuration stability.
        """
        inst = MedianBlurPerturber(ksize=ksize)
        for i in configuration_test_helper(inst):
            assert i.ksize == ksize

    @pytest.mark.parametrize("kwargs, expectation", [
        ({"ksize": 5}, does_not_raise()),
        ({"ksize": 4}, pytest.raises(ValueError, match=r"MedianBlurPerturber invalid ksize")),
        ({"ksize": 3}, does_not_raise()),
        ({"ksize": 2}, pytest.raises(ValueError, match=r"MedianBlurPerturber invalid ksize")),
        ({"ksize": 1}, pytest.raises(ValueError, match=r"MedianBlurPerturber invalid ksize")),
    ])
    def test_configuration_bounds(self, kwargs: Dict[str, Any], expectation: ContextManager) -> None:
        """
        Test that an exception is properly raised (or not) based on argument value.
        """
        with expectation:
            MedianBlurPerturber(**kwargs)


EXPECTED_AVERAGE = np.array([[4, 4, 4], [5, 5, 5], [6, 6, 6]], dtype=np.uint8)
EXPECTED_GAUSSIAN = np.array([[3, 4, 4], [5, 5, 6], [6, 7, 7]], dtype=np.uint8)
EXPECTED_MEDIAN = np.array([[2, 3, 3], [4, 5, 6], [7, 7, 8]], dtype=np.uint8)
