from collections.abc import Hashable, Iterable

import numpy as np
import pytest
from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.utils._nop_perturber import _NOPPerturber
from tests.impls.perturb.test_perturber_utils import perturber_assertions

rng = np.random.default_rng()


@pytest.mark.parametrize(
    "image",
    [
        rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8),
        np.ones((256, 256, 3), dtype=np.float32),
    ],
)
def test_perturber_assertions(image: np.ndarray) -> None:
    """Run on a dummy image to ensure output matches expectations."""
    inst = _NOPPerturber()

    # Test perturb interface directly
    perturber_assertions(perturb=inst.perturb, image=image, expected=image)

    # Test callable
    perturber_assertions(perturb=inst, image=image, expected=image)


def test_config() -> None:
    """Test configuration stability."""
    inst = _NOPPerturber()
    configuration_test_helper(inst)


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
def test_perturb_with_boxes(boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]) -> None:
    """Test that bounding boxes do not change during perturb."""
    inst = _NOPPerturber()
    _, out_boxes = inst.perturb(image=np.ones((256, 256, 3)), boxes=boxes)
    assert boxes == out_boxes
