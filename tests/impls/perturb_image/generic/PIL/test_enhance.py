import unittest.mock as mock
from collections.abc import Hashable, Iterable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image
from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from syrupy.assertion import SnapshotAssertion

from nrtk.impls.perturb_image.generic.PIL.enhance import (
    BrightnessPerturber,
    ColorPerturber,
    ContrastPerturber,
    SharpnessPerturber,
)
from nrtk.utils._exceptions import PillowImportError
from tests.impls.perturb_image.test_perturber_utils import perturber_assertions
from tests.impls.test_pybsm_utils import TIFFImageSnapshotExtension

rng = np.random.default_rng()

INPUT_IMG_FILE_PATH = "./docs/examples/maite/data/visdrone_img.jpg"


@pytest.fixture
def tiff_snapshot(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(TIFFImageSnapshotExtension)


@pytest.mark.skipif(not BrightnessPerturber.is_usable(), reason=str(PillowImportError()))
class TestBrightnessPerturber:
    def test_consistency(self, tiff_snapshot: SnapshotAssertion) -> None:
        """Run on a real image to ensure output matches."""
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        out_img = perturber_assertions(
            perturb=BrightnessPerturber(factor=0.2),
            image=image,
        )
        tiff_snapshot.assert_match(out_img)

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
        inst = BrightnessPerturber(factor=0.5)
        _, out_boxes = inst.perturb(np.ones((256, 256, 3)), boxes=boxes)
        assert boxes == out_boxes


@pytest.mark.skipif(not ColorPerturber.is_usable(), reason=str(PillowImportError()))
class TestColorPerturber:
    def test_consistency(self, tiff_snapshot: SnapshotAssertion) -> None:
        """Run on a real image to ensure output matches."""
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        factor = 0.2
        # Test callable
        out_img = perturber_assertions(
            perturb=ColorPerturber(factor=factor),
            image=image,
        )
        tiff_snapshot.assert_match(out_img)

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
        inst = ColorPerturber(factor=0.5)
        _, out_boxes = inst.perturb(np.ones((256, 256, 3)), boxes=boxes)
        assert boxes == out_boxes


@pytest.mark.skipif(not ContrastPerturber.is_usable(), reason=str(PillowImportError()))
class TestContrastPerturber:
    def test_consistency(self, tiff_snapshot: SnapshotAssertion) -> None:
        """Run on a real image to ensure output matches."""
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        factor = 0.2

        # Test callable
        out_img = perturber_assertions(
            perturb=ContrastPerturber(factor=factor),
            image=image,
        )
        tiff_snapshot.assert_match(out_img)

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
        inst = ContrastPerturber(factor=0.5)
        _, out_boxes = inst.perturb(np.ones((256, 256, 3)), boxes=boxes)
        assert boxes == out_boxes


@pytest.mark.skipif(not SharpnessPerturber.is_usable(), reason=str(PillowImportError()))
class TestSharpnessPerturber:
    def test_consistency(self, tiff_snapshot: SnapshotAssertion) -> None:
        """Run on a real image to ensure output matches."""
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        factor = 0.2

        # Test callable
        out_img = perturber_assertions(
            perturb=SharpnessPerturber(factor=factor),
            image=image,
        )
        tiff_snapshot.assert_match(out_img)

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
        inst = SharpnessPerturber(factor=0.5)
        _, out_boxes = inst.perturb(np.ones((256, 256, 3)), boxes=boxes)
        assert boxes == out_boxes


@mock.patch.object(BrightnessPerturber, "is_usable")
def test_missing_deps_brightness_perturber(mock_is_usable: MagicMock) -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    mock_is_usable.return_value = False
    assert not BrightnessPerturber.is_usable()
    with pytest.raises(PillowImportError):
        BrightnessPerturber()


@mock.patch.object(ColorPerturber, "is_usable")
def test_missing_deps_color_perturber(mock_is_usable: MagicMock) -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    mock_is_usable.return_value = False
    assert not ColorPerturber.is_usable()
    with pytest.raises(PillowImportError):
        ColorPerturber()


@mock.patch.object(ContrastPerturber, "is_usable")
def test_missing_deps_contrast_perturber(mock_is_usable: MagicMock) -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    mock_is_usable.return_value = False
    assert not ContrastPerturber.is_usable()
    with pytest.raises(PillowImportError):
        ContrastPerturber()


@mock.patch.object(SharpnessPerturber, "is_usable")
def test_missing_deps_sharpness_perturber(mock_is_usable: MagicMock) -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    mock_is_usable.return_value = False
    assert not SharpnessPerturber.is_usable()
    with pytest.raises(PillowImportError):
        SharpnessPerturber()
