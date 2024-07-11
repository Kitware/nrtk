import random
from contextlib import nullcontext as does_not_raise
from typing import ContextManager, Dict, Hashable, Sequence, Tuple

import numpy as np
import pytest
from PIL import Image
from pybsm.otf import dark_current_from_density
from smqtk_core.configuration import configuration_test_helper
from smqtk_detection.impls.detect_image_objects.random_detector import RandomDetector
from smqtk_image_io import AxisAlignedBoundingBox

from nrtk.impls.gen_object_detector_blackbox_response.simple_pybsm_generator import (
    SimplePybsmGenerator,
)
from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
from nrtk.impls.perturb_image_factory.pybsm import CustomPybsmPerturbImageFactory
from nrtk.impls.score_detections.random_scorer import RandomScorer

from .test_generator_utils import gen_rand_dets, generator_assertions

INPUT_IMG_FILE = (
    "./examples/pybsm/data/M-41 Walker Bulldog (USA) width 319cm height 272cm.tiff"
)


def create_sample_sensor() -> PybsmSensor:

    name = "L32511x"

    # telescope focal length (m)
    f = 4
    # Telescope diameter (m)
    D = 275e-3  # noqa: N806

    # detector pitch (m)
    p = 0.008e-3

    # Optical system transmission, red  band first (m)
    opt_trans_wavelengths = np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6
    # guess at the full system optical transmission (excluding obscuration)
    optics_transmission = 0.5 * np.ones(opt_trans_wavelengths.shape[0])

    # Relative linear telescope obscuration
    eta = 0.4  # guess

    # detector width is assumed to be equal to the pitch
    w_x = p
    w_y = p
    # integration time (s) - this is a maximum, the actual integration time will be
    # determined by the well fill percentage
    int_time = 30.0e-3

    # dark current density of 1 nA/cm2 guess, guess mid range for a silicon camera
    dark_current = dark_current_from_density(1e-5, w_x, w_y)

    # rms read noise (rms electrons)
    read_noise = 25.0

    # maximum ADC level (electrons)
    max_n = 96000.0

    # bit depth
    bitdepth = 11.9

    # maximum allowable well fill (see the paper for the logic behind this)
    max_well_fill = 0.6

    # jitter (radians) - The Olson paper says that its "good" so we'll guess 1/4 ifov rms
    s_x = 0.25 * p / f
    s_y = s_x

    # drift (radians/s) - again, we'll guess that it's really good
    da_x = 100e-6
    da_y = da_x

    # etector quantum efficiency as a function of wavelength (microns)
    # for a generic high quality back-illuminated silicon array
    # https://www.photometrics.com/resources/learningzone/quantumefficiency.php
    qe_wavelengths = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]) * 1.0e-6
    qe = np.array([0.05, 0.6, 0.75, 0.85, 0.85, 0.75, 0.5, 0.2, 0])

    sensor = PybsmSensor(
        name,
        D,
        f,
        p,
        opt_trans_wavelengths,
        optics_transmission,
        eta,
        w_x,
        w_y,
        int_time,
        dark_current,
        read_noise,
        max_n,
        bitdepth,
        max_well_fill,
        s_x,
        s_y,
        da_x,
        da_y,
        qe_wavelengths,
        qe,
    )

    return sensor


def create_sample_scenario() -> PybsmScenario:
    altitude = 9000.0
    # range to target
    ground_range = 60000.0

    scenario_name = "niceday"
    # weather model
    ihaze = 1
    scenario = PybsmScenario(scenario_name, ihaze, altitude, ground_range)
    scenario.aircraft_speed = 100.0

    return scenario


class TestSimplePyBSMGenerator:
    @pytest.mark.parametrize(
        ("images", "img_gsds", "ground_truth", "expectation"),
        [
            (
                [
                    np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                    for _ in range(4)
                ],
                np.random.rand(4, 1),
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
                np.random.rand(2, 1),
                [gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11))],
                pytest.raises(ValueError, match=r"Size mismatch."),
            ),
            (
                [
                    np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                    for _ in range(2)
                ],
                np.random.rand(4, 1),
                [
                    gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11))
                    for _ in range(2)
                ],
                pytest.raises(ValueError, match=r"Size mismatch."),
            ),
        ],
    )
    def test_configuration(
        self,
        images: Sequence[np.ndarray],
        img_gsds: Sequence[float],
        ground_truth: Sequence[
            Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]
        ],
        expectation: ContextManager,
    ) -> None:
        """Test configuration stability."""
        with expectation:
            inst = SimplePybsmGenerator(
                images=images, img_gsds=img_gsds, ground_truth=ground_truth
            )

            for i in configuration_test_helper(inst):
                assert i.images == images
                np.testing.assert_equal(i.img_gsds, img_gsds)
                assert i.ground_truth == ground_truth

    @pytest.mark.parametrize(
        ("images", "img_gsds", "ground_truth", "idx", "expectation"),
        [
            (
                [
                    np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                    for _ in range(3)
                ],
                np.random.rand(3, 1),
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
                np.random.rand(3, 1),
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
                np.random.rand(3, 1),
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
                np.random.rand(3, 1),
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
                np.random.rand(3, 1),
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
        img_gsds: Sequence[float],
        ground_truth: Sequence[
            Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]
        ],
        idx: int,
        expectation: ContextManager,
    ) -> None:
        """Ensure it is possible to index the generator and that len(generator) matches expectations."""
        inst = SimplePybsmGenerator(
            images=images, img_gsds=img_gsds, ground_truth=ground_truth
        )

        assert len(inst) == len(images)
        with expectation:
            inst_im, inst_gt, extra = inst[idx]
            assert np.array_equal(inst_im, images[idx])
            assert inst_gt == ground_truth[idx]
            assert extra["img_gsd"] == img_gsds[idx]

    @pytest.mark.parametrize(
        ("images", "img_gsds", "ground_truth", "perturber_factories", "verbose"),
        [
            (
                [np.array(Image.open(INPUT_IMG_FILE)) for _ in range(5)],
                [3.19 / 160.0 for _ in range(5)],
                [
                    gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11))
                    for _ in range(5)
                ],
                [
                    CustomPybsmPerturbImageFactory(
                        sensor=create_sample_sensor(),
                        scenario=create_sample_scenario(),
                        theta_keys=["ground_range"],
                        thetas=[[10000, 20000]],
                    ),
                    CustomPybsmPerturbImageFactory(
                        sensor=create_sample_sensor(),
                        scenario=create_sample_scenario(),
                        theta_keys=["ground_range"],
                        thetas=[[20000, 30000]],
                    ),
                ],
                False,
            ),
            (
                [np.array(Image.open(INPUT_IMG_FILE)) for _ in range(2)],
                [3.19 / 160.0 for _ in range(2)],
                [
                    gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11))
                    for _ in range(2)
                ],
                [
                    CustomPybsmPerturbImageFactory(
                        sensor=create_sample_sensor(),
                        scenario=create_sample_scenario(),
                        theta_keys=["altitude", "ground_range"],
                        thetas=[[2000, 3000], [10000, 20000]],
                    )
                ],
                True,
            ),
        ],
    )
    def test_generate(
        self,
        images: Sequence[np.ndarray],
        img_gsds: Sequence[float],
        ground_truth: Sequence[
            Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]
        ],
        perturber_factories: Sequence[CustomPybsmPerturbImageFactory],
        verbose: bool,
    ) -> None:
        """Ensure generation assertions hold."""
        inst = SimplePybsmGenerator(
            images=images, img_gsds=img_gsds, ground_truth=ground_truth
        )

        generator_assertions(
            generator=inst,
            perturber_factories=perturber_factories,
            detector=RandomDetector(),
            scorer=RandomScorer(),
            batch_size=1,
            verbose=verbose,
        )
