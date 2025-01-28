from collections.abc import Hashable, Sequence
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any

import numpy as np
import pytest
from PIL import Image
from smqtk_core.configuration import configuration_test_helper
from smqtk_detection.impls.detect_image_objects.random_detector import RandomDetector
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.impls.gen_object_detector_blackbox_response.simple_pybsm_generator import (
    SimplePybsmGenerator,
)
from nrtk.impls.perturb_image_factory.pybsm import CustomPybsmPerturbImageFactory
from nrtk.impls.score_detections.random_scorer import RandomScorer
from tests.impls.gen_object_detector_blackbox_response.test_generator_utils import gen_rand_dets, generator_assertions
from tests.impls.test_pybsm_utils import create_sample_scenario, create_sample_sensor

INPUT_IMG_FILE = "./examples/pybsm/data/M-41 Walker Bulldog (USA) width 319cm height 272cm.tiff"

rng = np.random.default_rng()


@pytest.mark.skipif(
    not SimplePybsmGenerator.is_usable(),
    reason="pyBSM with OpenCV not found. Please install 'nrtk[pybsm-graphics]' or `nrtk[pybsm-headless]`.",
)
class TestSimplePyBSMGenerator:
    @pytest.mark.parametrize(
        ("images", "img_gsds", "ground_truth", "expectation"),
        [
            (
                [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(4)],
                rng.random((4, 1)),
                [gen_rand_dets(im_shape=(256, 256), n_dets=rng.integers(1, 11)) for _ in range(4)],
                does_not_raise(),
            ),
            (
                [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(2)],
                rng.random((2, 1)),
                [gen_rand_dets(im_shape=(256, 256), n_dets=rng.integers(1, 11))],
                pytest.raises(ValueError, match=r"Size mismatch."),
            ),
            (
                [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(2)],
                rng.random((4, 1)),
                [gen_rand_dets(im_shape=(256, 256), n_dets=rng.integers(1, 11)) for _ in range(2)],
                pytest.raises(ValueError, match=r"Size mismatch."),
            ),
        ],
    )
    def test_configuration(
        self,
        images: Sequence[np.ndarray],
        img_gsds: Sequence[float],
        ground_truth: Sequence[Sequence[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]],
        expectation: AbstractContextManager,
    ) -> None:
        """Test configuration stability."""
        with expectation:
            inst = SimplePybsmGenerator(images=images, img_gsds=img_gsds, ground_truth=ground_truth)

            for i in configuration_test_helper(inst):
                assert i.images == images
                np.testing.assert_equal(i.img_gsds, img_gsds)
                assert i.ground_truth == ground_truth

    @pytest.mark.parametrize(
        ("images", "img_gsds", "ground_truth", "idx", "expectation"),
        [
            (
                [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(3)],
                rng.random((3, 1)),
                [gen_rand_dets(im_shape=(256, 256), n_dets=rng.integers(1, 11)) for _ in range(3)],
                0,
                does_not_raise(),
            ),
            (
                [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(3)],
                rng.random((3, 1)),
                [gen_rand_dets(im_shape=(256, 256), n_dets=rng.integers(1, 11)) for _ in range(3)],
                1,
                does_not_raise(),
            ),
            (
                [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(3)],
                rng.random((3, 1)),
                [gen_rand_dets(im_shape=(256, 256), n_dets=rng.integers(1, 11)) for _ in range(3)],
                2,
                does_not_raise(),
            ),
            (
                [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(3)],
                rng.random((3, 1)),
                [gen_rand_dets(im_shape=(256, 256), n_dets=rng.integers(1, 11)) for _ in range(3)],
                -1,
                pytest.raises(IndexError),
            ),
            (
                [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(3)],
                rng.random((3, 1)),
                [gen_rand_dets(im_shape=(256, 256), n_dets=rng.integers(1, 11)) for _ in range(3)],
                5,
                pytest.raises(IndexError),
            ),
        ],
    )
    def test_indexing(
        self,
        images: Sequence[np.ndarray],
        img_gsds: Sequence[float],
        ground_truth: Sequence[Sequence[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]],
        idx: int,
        expectation: AbstractContextManager,
    ) -> None:
        """Ensure it is possible to index the generator and that len(generator) matches expectations."""
        inst = SimplePybsmGenerator(images=images, img_gsds=img_gsds, ground_truth=ground_truth)

        assert len(inst) == len(images)
        with expectation:
            inst_im, inst_gt, extra = inst[idx]
            assert np.array_equal(inst_im, images[idx])
            assert inst_gt == ground_truth[idx]
            assert extra["img_gsd"] == img_gsds[idx]

    @pytest.mark.parametrize(
        ("images", "img_gsds", "ground_truth", "perturber_factory_configs", "verbose"),
        [
            (
                [np.array(Image.open(INPUT_IMG_FILE)) for _ in range(5)],
                [3.19 / 160.0 for _ in range(5)],
                [gen_rand_dets(im_shape=(256, 256), n_dets=rng.integers(1, 11)) for _ in range(5)],
                [
                    {
                        "sensor": create_sample_sensor(),
                        "scenario": create_sample_scenario(),
                        "theta_keys": ["ground_range"],
                        "thetas": [[10000, 20000]],
                    },
                    {
                        "sensor": create_sample_sensor(),
                        "scenario": create_sample_scenario(),
                        "theta_keys": ["ground_range"],
                        "thetas": [[20000, 30000]],
                    },
                ],
                False,
            ),
            (
                [np.array(Image.open(INPUT_IMG_FILE)) for _ in range(2)],
                [3.19 / 160.0 for _ in range(2)],
                [gen_rand_dets(im_shape=(256, 256), n_dets=rng.integers(1, 11)) for _ in range(2)],
                [
                    {
                        "sensor": create_sample_sensor(),
                        "scenario": create_sample_scenario(),
                        "theta_keys": ["altitude", "ground_range"],
                        "thetas": [[2000, 3000], [10000, 20000]],
                    },
                ],
                True,
            ),
        ],
    )
    def test_generate(
        self,
        images: Sequence[np.ndarray],
        img_gsds: Sequence[float],
        ground_truth: Sequence[Sequence[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]],
        perturber_factory_configs: Sequence[dict[str, Any]],
        verbose: bool,
    ) -> None:
        """Ensure generation assertions hold."""
        perturber_factories = [
            CustomPybsmPerturbImageFactory(**perturber_factory_config)
            for perturber_factory_config in perturber_factory_configs
        ]
        inst = SimplePybsmGenerator(images=images, img_gsds=img_gsds, ground_truth=ground_truth)

        generator_assertions(
            generator=inst,
            perturber_factories=perturber_factories,
            detector=RandomDetector(),
            scorer=RandomScorer(),
            batch_size=1,
            verbose=verbose,
        )
