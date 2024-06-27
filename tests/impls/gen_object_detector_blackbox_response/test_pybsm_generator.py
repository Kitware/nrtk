import numpy as np
import pytest
import random
from pybsm.otf import dark_current_from_density
from PIL import Image
from contextlib import nullcontext as does_not_raise
from typing import ContextManager, Dict, Hashable, Sequence, Tuple

from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io import AxisAlignedBoundingBox
from smqtk_detection.impls.detect_image_objects.random_detector import RandomDetector

from nrtk.impls.gen_object_detector_blackbox_response.simple_pybsm_generator import SimplePybsmGenerator
from nrtk.impls.perturb_image_factory.pybsm import CustomPybsmPerturbImageFactory
from nrtk.impls.score_detections.random_scorer import RandomScorer
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario

from .test_generator_utils import generator_assertions, gen_rand_dets

INPUT_IMG_FILE = './examples/pybsm/data/M-41 Walker Bulldog (USA) width 319cm height 272cm.tiff'


def createSampleSensor() -> PybsmSensor:

    name = 'L32511x'

    # telescope focal length (m)
    f = 4
    # Telescope diameter (m)
    D = 275e-3

    # detector pitch (m)
    p = .008e-3

    # Optical system transmission, red  band first (m)
    optTransWavelengths = np.array([0.58-.08, 0.58+.08])*1.0e-6
    # guess at the full system optical transmission (excluding obscuration)
    opticsTransmission = 0.5*np.ones(optTransWavelengths.shape[0])

    # Relative linear telescope obscuration
    eta = 0.4  # guess

    # detector width is assumed to be equal to the pitch
    wx = p
    wy = p
    # integration time (s) - this is a maximum, the actual integration time will be
    # determined by the well fill percentage
    intTime = 30.0e-3

    # dark current density of 1 nA/cm2 guess, guess mid range for a silicon camera
    darkCurrent = dark_current_from_density(1e-5, wx, wy)

    # rms read noise (rms electrons)
    readNoise = 25.0

    # maximum ADC level (electrons)
    maxN = 96000.0

    # bit depth
    bitdepth = 11.9

    # maximum allowable well fill (see the paper for the logic behind this)
    maxWellFill = .6

    # jitter (radians) - The Olson paper says that its "good" so we'll guess 1/4 ifov rms
    sx = 0.25*p/f
    sy = sx

    # drift (radians/s) - again, we'll guess that it's really good
    dax = 100e-6
    day = dax

    # etector quantum efficiency as a function of wavelength (microns)
    # for a generic high quality back-illuminated silicon array
    # https://www.photometrics.com/resources/learningzone/quantumefficiency.php
    qewavelengths = np.array([.3, .4, .5, .6, .7, .8, .9, 1.0, 1.1])*1.0e-6
    qe = np.array([0.05, 0.6, 0.75, 0.85, .85, .75, .5, .2, 0])

    sensor = PybsmSensor(name, D, f, p, optTransWavelengths,
                         opticsTransmission, eta, wx, wy,
                         intTime, darkCurrent, readNoise,
                         maxN, bitdepth, maxWellFill, sx, sy,
                         dax, day, qewavelengths, qe)

    return sensor


def createSampleScenario() -> PybsmScenario:
    altitude = 9000.0
    # range to target
    groundRange = 60000.0

    scenario_name = 'niceday'
    # weather model
    ihaze = 1
    scenario = PybsmScenario(scenario_name, ihaze, altitude, groundRange)
    scenario.aircraft_speed = 100.0

    return scenario


class TestSimplePyBSMGenerator:
    @pytest.mark.parametrize("images, img_gsds, groundtruth, expectation", [
        ([np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(4)],
            np.random.rand(4, 1),
            [gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11)) for _ in range(4)],
            does_not_raise()),
        ([np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(2)],
            np.random.rand(2, 1),
            [gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11))],
            pytest.raises(ValueError, match=r"Size mismatch.")),
        ([np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(2)],
            np.random.rand(4, 1),
            [gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11)) for _ in range(2)],
            pytest.raises(ValueError, match=r"Size mismatch."))

    ])
    def test_configuration(
        self,
        images: Sequence[np.ndarray],
        img_gsds: Sequence[float],
        groundtruth: Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]],
        expectation: ContextManager
    ) -> None:
        """
        Test configuration stability.
        """
        with expectation:
            inst = SimplePybsmGenerator(
                images=images,
                imggsds=img_gsds,
                groundtruth=groundtruth
            )

            for i in configuration_test_helper(inst):
                assert i.images == images
                np.testing.assert_equal(i.imggsds, img_gsds)
                assert i.groundtruth == groundtruth

    @pytest.mark.parametrize("images, img_gsds, groundtruth, idx, expectation", [
        ([np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(3)],
            np.random.rand(3, 1),
            [gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11)) for _ in range(3)],
            0, does_not_raise()),
        ([np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(3)],
            np.random.rand(3, 1),
            [gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11)) for _ in range(3)],
            1, does_not_raise()),
        ([np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(3)],
            np.random.rand(3, 1),
            [gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11)) for _ in range(3)],
            2, does_not_raise()),
        ([np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(3)],
            np.random.rand(3, 1),
            [gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11)) for _ in range(3)],
            -1, pytest.raises(IndexError)),
        ([np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(3)],
            np.random.rand(3, 1),
            [gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11)) for _ in range(3)],
            5, pytest.raises(IndexError)),
    ])
    def test_indexing(
        self,
        images: Sequence[np.ndarray],
        img_gsds: Sequence[float],
        groundtruth: Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]],
        idx: int,
        expectation: ContextManager
    ) -> None:
        """
        Ensure it is possible to index the generator and that len(generator) matches expectations.
        """

        inst = SimplePybsmGenerator(
            images=images,
            imggsds=img_gsds,
            groundtruth=groundtruth
        )

        assert len(inst) == len(images)
        with expectation:
            inst_im, inst_gt, extra = inst[idx]
            assert np.array_equal(inst_im, images[idx])
            assert inst_gt == groundtruth[idx]
            assert extra['img_gsd'] == img_gsds[idx]

    @pytest.mark.parametrize("images, img_gsds, groundtruth, perturber_factories, verbose", [
        ([np.array(Image.open(INPUT_IMG_FILE)) for _ in range(5)],
            [3.19/160.0 for _ in range(5)],
            [gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11)) for _ in range(5)],
            [CustomPybsmPerturbImageFactory(sensor=createSampleSensor(), scenario=createSampleScenario(),
                                            theta_keys=["groundRange"], thetas=[[10000, 20000]]),
                CustomPybsmPerturbImageFactory(sensor=createSampleSensor(), scenario=createSampleScenario(),
                                               theta_keys=["groundRange"], thetas=[[20000, 30000]])],
            False),
        ([np.array(Image.open(INPUT_IMG_FILE)) for _ in range(2)],
            [3.19/160.0 for _ in range(2)],
            [gen_rand_dets(im_shape=(256, 256), n_dets=random.randint(1, 11)) for _ in range(2)],
            [CustomPybsmPerturbImageFactory(sensor=createSampleSensor(), scenario=createSampleScenario(),
                                            theta_keys=["altitude", "groundRange"], thetas=[[2000, 3000],
                                            [10000, 20000]])],
            True)
    ])
    def test_generate(
        self,
        images: Sequence[np.ndarray],
        img_gsds: Sequence[float],
        groundtruth: Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]],
        perturber_factories: Sequence[CustomPybsmPerturbImageFactory],
        verbose: bool
    ) -> None:
        """
        Ensure generation assertions hold.
        """

        inst = SimplePybsmGenerator(
            images=images,
            imggsds=img_gsds,
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
