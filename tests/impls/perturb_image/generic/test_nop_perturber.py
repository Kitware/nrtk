import numpy as np
import pytest
from smqtk_core.configuration import configuration_test_helper

from nrtk.impls.perturb_image.generic.nop_perturber import NOPPerturber
from tests.impls.perturb_image.test_perturber_utils import perturber_assertions

rng = np.random.default_rng()


@pytest.mark.parametrize(
    "image",
    [
        rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
        np.ones((256, 256, 3), dtype=np.float32),
    ],
)
def test_perturber_assertions(image: np.ndarray) -> None:
    """Run on a dummy image to ensure output matches expectations."""
    inst = NOPPerturber()

    # Test perturb interface directly
    perturber_assertions(perturb=inst.perturb, image=image, expected=image)

    # Test callable
    perturber_assertions(perturb=inst, image=image, expected=image)


def test_config() -> None:
    """Test configuration stability."""
    inst = NOPPerturber()
    configuration_test_helper(inst)
