from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any

import numpy as np
import pytest
from smqtk_core.configuration import configuration_test_helper

from nrtk.impls.perturb_image.generic.PIL.enhance import (
    BrightnessPerturber,
    ColorPerturber,
    ContrastPerturber,
    SharpnessPerturber,
)
from tests.impls.perturb_image.test_perturber_utils import perturber_assertions

rng = np.random.default_rng()


class TestBrightnessPerturber:
    def test_consistency(self) -> None:
        """Run on a dummy image to ensure output matches precomputed results."""
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        factor = 0.2

        # Test perturb interface directly
        inst = BrightnessPerturber(factor=factor)
        perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=EXPECTED_BRIGHTNESS,
        )

        # Test callable
        perturber_assertions(
            perturb=BrightnessPerturber(factor=factor),
            image=image,
            expected=EXPECTED_BRIGHTNESS,
        )

    @pytest.mark.parametrize(
        ("image", "factor"),
        [
            (rng.integers(0, 255, (256, 256, 3), dtype=np.uint8), 0.5),
            (np.ones((256, 256, 3), dtype=np.float32), 1.3),
            (np.ones((256, 256, 3), dtype=np.float64), 0.2),
        ],
    )
    def test_reproducibility(self, image: np.ndarray, factor: float) -> None:
        """Ensure results are reproducible."""
        # Test perturb interface directly
        inst = BrightnessPerturber(factor=factor)
        out_image = perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=None,
        )
        perturber_assertions(perturb=inst.perturb, image=image, expected=out_image)

        # Test callable
        perturber_assertions(perturb=inst, image=image, expected=out_image)

    @pytest.mark.parametrize("factor", [3.14, 0.5])
    def test_configuration(self, factor: float) -> None:
        """Test configuration stability."""
        inst = BrightnessPerturber(factor=factor)
        for i in configuration_test_helper(inst):
            assert i.factor == factor

    @pytest.mark.parametrize(
        ("kwargs", "expectation"),
        [
            ({"factor": 5}, does_not_raise()),
            ({"factor": 0.0}, does_not_raise()),
            (
                {"factor": -1.2},
                pytest.raises(ValueError, match=r"BrightnessPerturber invalid factor"),
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
            BrightnessPerturber(**kwargs)


class TestColorPerturber:
    def test_consistency(self) -> None:
        """Run on a dummy image to ensure output matches precomputed results."""
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        factor = 0.2

        # Test perturb interface directly
        inst = ColorPerturber(factor=factor)
        perturber_assertions(perturb=inst.perturb, image=image, expected=EXPECTED_COLOR)

        # Test callable
        perturber_assertions(
            perturb=ColorPerturber(factor=factor),
            image=image,
            expected=EXPECTED_COLOR,
        )

    @pytest.mark.parametrize(
        ("image", "factor"),
        [
            (rng.integers(0, 255, (256, 256, 3), dtype=np.uint8), 0.5),
            (np.ones((256, 256, 3), dtype=np.float32), 1.3),
            (np.ones((256, 256, 3), dtype=np.float64), 0.2),
        ],
    )
    def test_reproducibility(self, image: np.ndarray, factor: float) -> None:
        """Ensure results are reproducible."""
        # Test perturb interface directly
        inst = ColorPerturber(factor=factor)
        out_image = perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=None,
        )
        perturber_assertions(perturb=inst.perturb, image=image, expected=out_image)

        # Test callable
        perturber_assertions(perturb=inst, image=image, expected=out_image)

    @pytest.mark.parametrize("factor", [3.14, 0.5])
    def test_configuration(self, factor: float) -> None:
        """Test configuration stability."""
        inst = ColorPerturber(factor=factor)
        for i in configuration_test_helper(inst):
            assert i.factor == factor

    @pytest.mark.parametrize(
        ("kwargs", "expectation"),
        [
            ({"factor": 5}, does_not_raise()),
            ({"factor": 0.0}, does_not_raise()),
            (
                {"factor": -1.2},
                pytest.raises(ValueError, match=r"ColorPerturber invalid factor"),
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
            ColorPerturber(**kwargs)


class TestContrastPerturber:
    def test_consistency(self) -> None:
        """Run on a dummy image to ensure output matches precomputed results."""
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        factor = 0.2

        # Test perturb interface directly
        inst = ContrastPerturber(factor=factor)
        perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=EXPECTED_CONTRAST,
        )

        # Test callable
        perturber_assertions(
            perturb=ContrastPerturber(factor=factor),
            image=image,
            expected=EXPECTED_CONTRAST,
        )

    @pytest.mark.parametrize(
        ("image", "factor"),
        [
            (rng.integers(0, 255, (256, 256, 3), dtype=np.uint8), 0.5),
            (np.ones((256, 256, 3), dtype=np.float32), 1.3),
            (np.ones((256, 256, 3), dtype=np.float64), 0.2),
        ],
    )
    def test_reproducibility(self, image: np.ndarray, factor: float) -> None:
        """Ensure results are reproducible."""
        # Test perturb interface directly
        inst = ContrastPerturber(factor=factor)
        out_image = perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=None,
        )
        perturber_assertions(perturb=inst.perturb, image=image, expected=out_image)

        # Test callable
        perturber_assertions(perturb=inst, image=image, expected=out_image)

    @pytest.mark.parametrize("factor", [3.14, 0.5])
    def test_configuration(self, factor: float) -> None:
        """Test configuration stability."""
        inst = ContrastPerturber(factor=factor)
        for i in configuration_test_helper(inst):
            assert i.factor == factor

    @pytest.mark.parametrize(
        ("kwargs", "expectation"),
        [
            ({"factor": 5}, does_not_raise()),
            ({"factor": 0.0}, does_not_raise()),
            (
                {"factor": -1.2},
                pytest.raises(ValueError, match=r"ContrastPerturber invalid factor"),
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
            ContrastPerturber(**kwargs)


class TestSharpnessPerturber:
    def test_consistency(self) -> None:
        """Run on a dummy image to ensure output matches precomputed results."""
        image = np.array([[100, 20, 30], [40, 50, 60], [70, 80, 255]], dtype=np.uint8)
        factor = 0.2

        # Test perturb interface directly
        inst = SharpnessPerturber(factor=factor)
        perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=EXPECTED_SHARPNESS,
        )

        # Test callable
        perturber_assertions(
            perturb=SharpnessPerturber(factor=factor),
            image=image,
            expected=EXPECTED_SHARPNESS,
        )

    @pytest.mark.parametrize(
        ("image", "factor"),
        [
            (rng.integers(0, 255, (256, 256, 3), dtype=np.uint8), 0.5),
            (np.ones((256, 256, 3), dtype=np.float32), 1.3),
            (np.ones((256, 256, 3), dtype=np.float64), 0.2),
        ],
    )
    def test_reproducibility(self, image: np.ndarray, factor: float) -> None:
        """Ensure results are reproducible."""
        # Test perturb interface directly
        inst = SharpnessPerturber(factor=factor)
        out_image = perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=None,
        )
        perturber_assertions(perturb=inst.perturb, image=image, expected=out_image)

        # Test callable
        perturber_assertions(perturb=inst, image=image, expected=out_image)

    @pytest.mark.parametrize("factor", [1.3, 0.5])
    def test_configuration(self, factor: float) -> None:
        """Test configuration stability."""
        inst = SharpnessPerturber(factor=factor)
        for i in configuration_test_helper(inst):
            assert i.factor == factor

    @pytest.mark.parametrize(
        ("kwargs", "expectation"),
        [
            (
                {"factor": 5},
                pytest.raises(
                    ValueError,
                    match=r"SharpnessPerturber invalid sharpness factor",
                ),
            ),
            ({"factor": 2.0}, does_not_raise()),
            ({"factor": 1.5}, does_not_raise()),
            ({"factor": 0.0}, does_not_raise()),
            (
                {"factor": -1.2},
                pytest.raises(
                    ValueError,
                    match=r"SharpnessPerturber invalid sharpness factor",
                ),
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
            SharpnessPerturber(**kwargs)


EXPECTED_BRIGHTNESS = np.array([[0, 0, 0], [0, 1, 1], [1, 1, 1]], dtype=np.uint8)
EXPECTED_COLOR = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
EXPECTED_CONTRAST = np.array([[4, 4, 4], [4, 5, 5], [5, 5, 5]], dtype=np.uint8)
EXPECTED_SHARPNESS = np.array(
    [[100, 20, 30], [40, 66, 60], [70, 80, 255]],
    dtype=np.uint8,
)
