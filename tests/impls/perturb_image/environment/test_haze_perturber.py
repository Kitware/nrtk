from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any

import numpy as np
import pytest
from PIL import Image
from smqtk_core.configuration import configuration_test_helper
from syrupy.assertion import SnapshotAssertion

from nrtk.impls.perturb_image.environment.haze_perturber import HazePerturber
from tests.impls import INPUT_TANK_IMG_FILE_PATH as INPUT_IMG_FILE_PATH
from tests.impls.perturb_image.test_perturber_utils import perturber_assertions

rng = np.random.default_rng()


class TestHazePerturber:
    @pytest.mark.parametrize(
        ("metadata"),
        [
            {},
            {"sky_color": [0.5, 0.5, 0.5]},
            {"sky_color": [1, 1, 1], "depth_map": np.ones((3, 3, 1)) * 0.5},
        ],
    )
    def test_default_consistency(
        self,
        metadata: dict[str, Any],
    ) -> None:
        """Run on a dummy image to ensure output matches precomputed results."""
        image = np.ones((3, 3, 3)).astype(np.uint8)

        # Test perturb interface directly
        inst = HazePerturber()
        out_image = perturber_assertions(perturb=inst.perturb, image=image, **metadata)

        # Test callable
        perturber_assertions(
            perturb=HazePerturber(),
            image=image,
            expected=out_image,
            **metadata,
        )

    @pytest.mark.parametrize(
        ("factor", "metadata"),
        [
            (2.0, {}),
            (2.0, {"sky_color": [0.5, 0.5, 0.5]}),
            (2.0, {"sky_color": [0.5, 0.5, 0.5], "depth_map": np.ones((3, 3, 1)) * 0.5}),
        ],
    )
    def test_consistency(
        self,
        factor: float,
        metadata: dict[str, Any],
    ) -> None:
        """Run on a dummy image to ensure output matches precomputed results."""
        image = np.ones((3, 3, 3)).astype(np.uint8)

        # Test perturb interface directly
        inst = HazePerturber(factor=factor)
        out_image = perturber_assertions(perturb=inst.perturb, image=image, **metadata)

        # Test callable
        perturber_assertions(
            perturb=HazePerturber(factor=factor),
            image=image,
            expected=out_image,
            **metadata,
        )

    def test_regression(self, tiff_snapshot: SnapshotAssertion) -> None:
        """Regression testing results to detect API changes."""
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        inst = HazePerturber()
        depth_map = np.zeros_like(image).astype(np.float64)
        depth_map[...] = np.linspace(start=1, stop=0.2, num=depth_map.shape[0])[..., None]
        out_img = perturber_assertions(
            perturb=inst.perturb,
            image=image,
            sky_color=[122],
            depth_map=depth_map,
        )
        tiff_snapshot.assert_match(out_img)

    @pytest.mark.parametrize(
        ("factor"),
        [
            1.0,
            2.0,
        ],
    )
    def test_configuration(
        self,
        factor: float,
    ) -> None:
        """Test configuration stability."""
        inst = HazePerturber(factor=factor)
        for i in configuration_test_helper(inst):
            assert i.factor == factor

    @pytest.mark.parametrize(
        ("metadata", "expectation"),
        [
            ({"sky_color": [1, 1, 1], "depth_map": np.ones((3, 3, 1))}, does_not_raise()),
            (
                {"sky_color": [1, 1, 1], "depth_map": np.ones((3, 3))},
                pytest.raises(ValueError, match=r"image dims \(3\) does not match depth_map dims \(2\)"),
            ),
            (
                {"sky_color": [1, 1], "depth_map": np.ones((3, 3, 1))},
                pytest.raises(ValueError, match=r"image bands \(3\) do not match sky_color bands \(2\)"),
            ),
        ],
    )
    def test_execution_bounds(
        self,
        metadata: dict,
        expectation: AbstractContextManager,
    ) -> None:
        """Raise appropriate errors for specific parameters."""
        image = np.ones((3, 3, 3))
        inst = HazePerturber()
        with expectation:
            inst(image=image, **metadata)
