import random
from contextlib import nullcontext as does_not_raise
from typing import ContextManager, Dict, Hashable, Sequence, Tuple

import numpy as np
import pytest
from smqtk_core.configuration import configuration_test_helper
from smqtk_detection.impls.detect_image_objects.random_detector import RandomDetector
from smqtk_image_io import AxisAlignedBoundingBox

from nrtk.impls.gen_object_detector_blackbox_response.simple_generic_generator import (
    SimpleGenericGenerator,
)
from nrtk.impls.perturb_image.generic.cv2.blur import (
    AverageBlurPerturber,
    MedianBlurPerturber,
)
from nrtk.impls.perturb_image_factory.generic.step import StepPerturbImageFactory
from nrtk.impls.score_detections.random_scorer import RandomScorer
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory

from .test_generator_utils import gen_rand_dets, generator_assertions


class TestSimpleGenerator:
    @pytest.mark.parametrize(
        ("images", "ground_truth", "expectation"),
        [
            (
                [
                    np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                    for _ in range(4)
                ],
                [
                    gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11))
                    for _ in range(4)
                ],
                does_not_raise(),
            ),
            (
                [
                    np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                    for _ in range(2)
                ],
                [gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11))],
                pytest.raises(ValueError, match=r"Size mismatch."),
            ),
        ],
    )
    def test_configuration(
        self,
        images: Sequence[np.ndarray],
        ground_truth: Sequence[
            Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]
        ],
        expectation: ContextManager,
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
                [
                    np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                    for _ in range(3)
                ],
                [
                    gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11))
                    for _ in range(3)
                ],
                0,
                does_not_raise(),
            ),
            (
                [
                    np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                    for _ in range(3)
                ],
                [
                    gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11))
                    for _ in range(3)
                ],
                1,
                does_not_raise(),
            ),
            (
                [
                    np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                    for _ in range(3)
                ],
                [
                    gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11))
                    for _ in range(3)
                ],
                2,
                does_not_raise(),
            ),
            (
                [
                    np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                    for _ in range(3)
                ],
                [
                    gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11))
                    for _ in range(3)
                ],
                -1,
                pytest.raises(IndexError),
            ),
            (
                [
                    np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                    for _ in range(3)
                ],
                [
                    gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11))
                    for _ in range(3)
                ],
                5,
                pytest.raises(IndexError),
            ),
        ],
    )
    def test_indexing(
        self,
        images: Sequence[np.ndarray],
        ground_truth: Sequence[
            Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]
        ],
        idx: int,
        expectation: ContextManager,
    ) -> None:
        """Ensure it is possible to index the generator and that len(generator) matches expectations."""
        inst = SimpleGenericGenerator(images=images, ground_truth=ground_truth)

        assert len(inst) == len(images)
        with expectation:
            inst_im, inst_gt, extra = inst[idx]
            assert np.array_equal(inst_im, images[idx])
            assert inst_gt == ground_truth[idx]

    @pytest.mark.parametrize(
        ("images", "ground_truth", "perturber_factories", "verbose"),
        [
            (
                [
                    np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                    for _ in range(5)
                ],
                [
                    gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11))
                    for _ in range(5)
                ],
                [
                    StepPerturbImageFactory(
                        perturber=AverageBlurPerturber,
                        theta_key="ksize",
                        start=1,
                        stop=4,
                    ),
                    StepPerturbImageFactory(
                        perturber=MedianBlurPerturber,
                        theta_key="ksize",
                        start=3,
                        stop=8,
                        step=2,
                    ),
                ],
                False,
            ),
            (
                [
                    np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                    for _ in range(2)
                ],
                [
                    gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11))
                    for _ in range(2)
                ],
                [
                    StepPerturbImageFactory(
                        perturber=AverageBlurPerturber,
                        theta_key="ksize",
                        start=1,
                        stop=6,
                    )
                ],
                True,
            ),
        ],
    )
    def test_generate(
        self,
        images: Sequence[np.ndarray],
        ground_truth: Sequence[
            Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]
        ],
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
