import numpy as np
import pytest
import random
from contextlib import nullcontext as does_not_raise
from typing import ContextManager, Dict, Hashable, Sequence, Tuple

from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io import AxisAlignedBoundingBox
from smqtk_detection.impls.detect_image_objects.random_detector import RandomDetector

from nrtk.impls.gen_object_detector_blackbox_response.simple_generator import SimpleGenerator
from nrtk.impls.perturb_image.cv2.blur import AverageBlurPerturber, MedianBlurPerturber
from nrtk.impls.perturb_image_factory.step import StepPerturbImageFactory
from nrtk.impls.score_detection.random_scorer import RandomScorer
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory

from .test_generator_utils import generator_assertions, gen_rand_dets


class TestSimpleGenerator:
    @pytest.mark.parametrize("images, groundtruth, expectation", [
        ([np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(4)],
            [gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11)) for _ in range(4)],
            does_not_raise()),
        ([np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(2)],
            [gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11))],
            pytest.raises(ValueError, match=r"Size mismatch."))
    ])
    def test_configuration(
        self,
        images: Sequence[np.ndarray],
        groundtruth: Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]],
        expectation: ContextManager
    ) -> None:
        """
        Test configuration stability.
        """
        with expectation:
            inst = SimpleGenerator(
                images=images,
                groundtruth=groundtruth
            )

            for i in configuration_test_helper(inst):
                assert i.images == images
                assert i.groundtruth == groundtruth

    @pytest.mark.parametrize("images, groundtruth, idx, expectation", [
        ([np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(3)],
            [gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11)) for _ in range(3)],
            0, does_not_raise()),
        ([np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(3)],
            [gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11)) for _ in range(3)],
            1, does_not_raise()),
        ([np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(3)],
            [gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11)) for _ in range(3)],
            2, does_not_raise()),
        ([np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(3)],
            [gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11)) for _ in range(3)],
            -1, pytest.raises(IndexError)),
        ([np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(3)],
            [gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11)) for _ in range(3)],
            5, pytest.raises(IndexError)),
    ])
    def test_indexing(
        self,
        images: Sequence[np.ndarray],
        groundtruth: Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]],
        idx: int,
        expectation: ContextManager
    ) -> None:
        """
        Ensure it is possible to index the generator and that len(generator) matches expectations.
        """

        inst = SimpleGenerator(
            images=images,
            groundtruth=groundtruth
        )

        assert len(inst) == len(images)
        with expectation:
            inst_im, inst_gt = inst[idx]
            assert np.array_equal(inst_im, images[idx])
            assert inst_gt == groundtruth[idx]

    @pytest.mark.parametrize("images, groundtruth, perturber_factories, verbose", [
        ([np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(5)],
            [gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11)) for _ in range(5)],
            [StepPerturbImageFactory(perturber=AverageBlurPerturber, theta_key="ksize", start=1, stop=4),
                StepPerturbImageFactory(perturber=MedianBlurPerturber, theta_key="ksize", start=3, stop=8, step=2)],
            False),
        ([np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(2)],
            [gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11)) for _ in range(2)],
            [StepPerturbImageFactory(perturber=AverageBlurPerturber, theta_key="ksize", start=1, stop=6)],
            True)
    ])
    def test_generate(
        self,
        images: Sequence[np.ndarray],
        groundtruth: Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]],
        perturber_factories: Sequence[PerturbImageFactory],
        verbose: bool
    ) -> None:
        """
        Ensure generation assertions hold.
        """

        inst = SimpleGenerator(
            images=images,
            groundtruth=groundtruth
        )

        generator_assertions(
            generator=inst,
            perturber_factories=perturber_factories,
            detector=RandomDetector(),
            scorer=RandomScorer(),
            batch_size=1,
            verbose=verbose
        )
