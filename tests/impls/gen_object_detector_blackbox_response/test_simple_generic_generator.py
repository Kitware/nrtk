from collections.abc import Hashable, Sequence
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from smqtk_core.configuration import configuration_test_helper
from smqtk_detection.impls.detect_image_objects.random_detector import RandomDetector
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.impls.gen_object_detector_blackbox_response.simple_generic_generator import (
    SimpleGenericGenerator,
)
from nrtk.impls.perturb_image.generic.PIL.enhance import (
    BrightnessPerturber,
    ContrastPerturber,
)
from nrtk.impls.perturb_image_factory.generic.step import StepPerturbImageFactory
from nrtk.impls.score_detections.random_scorer import RandomScorer
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory

from .test_generator_utils import gen_rand_dets, generator_assertions

rng = np.random.default_rng()


class TestSimpleGenerator:
    @pytest.mark.parametrize(
        ("images", "ground_truth", "expectation"),
        [
            (
                [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(4)],
                [gen_rand_dets(im_shape=(256, 256), n_dets=rng.integers(1, 11)) for _ in range(4)],
                does_not_raise(),
            ),
            (
                [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(2)],
                [gen_rand_dets(im_shape=(256, 256), n_dets=rng.integers(1, 11))],
                pytest.raises(ValueError, match=r"Size mismatch."),
            ),
        ],
    )
    def test_configuration(
        self,
        images: Sequence[np.ndarray],
        ground_truth: Sequence[Sequence[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]],
        expectation: AbstractContextManager,
    ) -> None:
        """Test configuration stability."""
        with expectation:
            inst = SimpleGenericGenerator(images=images, ground_truth=ground_truth)

            for i in configuration_test_helper(inst):
                assert i.images == images
                assert i.ground_truth == ground_truth

    @pytest.mark.parametrize(
        ("images", "ground_truth", "idx", "expectation"),
        [
            (
                [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(3)],
                [gen_rand_dets(im_shape=(256, 256), n_dets=rng.integers(1, 11)) for _ in range(3)],
                0,
                does_not_raise(),
            ),
            (
                [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(3)],
                [gen_rand_dets(im_shape=(256, 256), n_dets=rng.integers(1, 11)) for _ in range(3)],
                1,
                does_not_raise(),
            ),
            (
                [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(3)],
                [gen_rand_dets(im_shape=(256, 256), n_dets=rng.integers(1, 11)) for _ in range(3)],
                2,
                does_not_raise(),
            ),
            (
                [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(3)],
                [gen_rand_dets(im_shape=(256, 256), n_dets=rng.integers(1, 11)) for _ in range(3)],
                -1,
                pytest.raises(IndexError),
            ),
            (
                [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(3)],
                [gen_rand_dets(im_shape=(256, 256), n_dets=rng.integers(1, 11)) for _ in range(3)],
                5,
                pytest.raises(IndexError),
            ),
        ],
    )
    def test_indexing(
        self,
        images: Sequence[np.ndarray],
        ground_truth: Sequence[Sequence[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]],
        idx: int,
        expectation: AbstractContextManager,
    ) -> None:
        """Ensure it is possible to index the generator and that len(generator) matches expectations."""
        inst = SimpleGenericGenerator(images=images, ground_truth=ground_truth)

        assert len(inst) == len(images)
        with expectation:
            inst_im, inst_gt, _ = inst[idx]
            assert np.array_equal(inst_im, images[idx])
            assert inst_gt == ground_truth[idx]

    @pytest.mark.parametrize(
        ("images", "ground_truth", "perturber_factories", "verbose"),
        [
            (
                [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(5)],
                [gen_rand_dets(im_shape=(256, 256), n_dets=rng.integers(1, 11)) for _ in range(5)],
                [
                    StepPerturbImageFactory(
                        perturber=BrightnessPerturber,
                        theta_key="factor",
                        start=1.0,
                        stop=4.0,
                        to_int=True,
                    ),
                    StepPerturbImageFactory(
                        perturber=ContrastPerturber,
                        theta_key="factor",
                        start=3.0,
                        stop=8.0,
                        step=2.0,
                        to_int=True,
                    ),
                ],
                False,
            ),
            (
                [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(2)],
                [gen_rand_dets(im_shape=(256, 256), n_dets=rng.integers(1, 11)) for _ in range(2)],
                [
                    StepPerturbImageFactory(
                        perturber=BrightnessPerturber,
                        theta_key="factor",
                        start=1,
                        stop=6,
                        to_int=True,
                    ),
                ],
                True,
            ),
        ],
    )
    def test_generate(
        self,
        images: Sequence[np.ndarray],
        ground_truth: Sequence[Sequence[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]],
        perturber_factories: Sequence[PerturbImageFactory],
        verbose: bool,
    ) -> None:
        """Ensure generation assertions hold."""
        inst = SimpleGenericGenerator(images=images, ground_truth=ground_truth)

        generator_assertions(
            generator=inst,
            perturber_factories=perturber_factories,
            detector=RandomDetector(),
            scorer=RandomScorer(),
            batch_size=1,
            verbose=verbose,
        )
