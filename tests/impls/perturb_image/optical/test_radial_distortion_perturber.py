from collections.abc import Sequence

import numpy as np
import pytest
from PIL import Image
from syrupy.assertion import SnapshotAssertion

from nrtk.impls.perturb_image.optical.radial_distortion_perturber import RadialDistortionPerturber
from tests.impls import INPUT_TANK_IMG_FILE_PATH as INPUT_IMG_FILE_PATH
from tests.impls.perturb_image.perturber_utils import perturber_assertions

rng = np.random.default_rng()


@pytest.mark.core
class TestRadialDistortionPerturber:
    @pytest.mark.parametrize(
        ("image"),
        [
            (rng.integers(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)),
            (np.ones((3, 3, 3)).astype(np.uint8)),
        ],
    )
    def test_consistency(
        self,
        image: np.ndarray,
    ) -> None:
        """Run perturber twice with consistent seed to ensure repeatable results."""
        k = [0.05, -0.01, 0.02]

        # Test perturb interface directly
        inst = RadialDistortionPerturber(k=k)
        out_image = perturber_assertions(perturb=inst.perturb, image=image)

        # Test callable
        perturber_assertions(
            perturb=RadialDistortionPerturber(k=k),
            image=image,
            expected=out_image,
        )

    @pytest.mark.parametrize(
        ("k"),
        [
            ([0, 0, 0]),
            ([0.1, 0.2, 0.3]),
            ([0.05, -0.01, 0.02]),
            ([-0.02, -0.05, 0]),
        ],
    )
    def test_regression(self, k: Sequence[float], psnr_tiff_snapshot: SnapshotAssertion) -> None:
        """Regression testing results to detect API changes."""
        grayscale_image = Image.open(INPUT_IMG_FILE_PATH)
        image = Image.new(mode="RGB", size=grayscale_image.size)
        image.paste(grayscale_image)
        image = np.array(image)
        inst = RadialDistortionPerturber(k=k)
        out_img = perturber_assertions(
            perturb=inst.perturb,
            image=image,
        )
        psnr_tiff_snapshot.assert_match(out_img)

    @pytest.mark.parametrize(
        ("color"),
        [
            ([0, 0, 0]),
            ([255, 0, 0]),
        ],
    )
    def test_color(self, color: Sequence[int]) -> None:
        image = np.ones((255, 255, 3)).astype(np.uint8)
        k = [0.5, 0.5, 0.5]
        inst = RadialDistortionPerturber(color_fill=color, k=k)
        out_img = inst.perturb(image=image)

        assert (out_img[0][0] == color).all()

    @pytest.mark.parametrize(
        ("k"),
        [
            ([],),
            ([1]),
            ([1, 1, 1, 1]),
        ],
    )
    def test_bad_k(self, k: Sequence[float]) -> None:
        with pytest.raises(ValueError, match="k must have exactly 3 values"):
            RadialDistortionPerturber(k=k)
