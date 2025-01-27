from collections.abc import Hashable, Sequence

import numpy as np
from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.interfaces.gen_object_detector_blackbox_response import (
    GenerateObjectDetectorBlackboxResponse,
)
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from nrtk.interfaces.score_detections import ScoreDetections

rng = np.random.default_rng()


def generator_assertions(
    generator: GenerateObjectDetectorBlackboxResponse,
    perturber_factories: Sequence[PerturbImageFactory],
    detector: DetectImageObjects,
    scorer: ScoreDetections,
    batch_size: int,
    verbose: bool,
) -> None:
    """Test some blanket assertions for generators.

    1) Length of curve response should be equal to the number of perturber combinations
    2) There should be an individual score per perturber combination per data pair

    :param generator: The generator object to test.
    :param perturber_factories: Sequence of perturber factories to perturb data.
    :param detector: Object detector to generate detections on perturbed data.
    :param scorer: Scorer to score resultant detections.
    :param batch_size: Number of images to predict upon at once.
    :param verbose: Control the verbosity of the output.
    """
    curve, full_scores = generator(
        blackbox_perturber_factories=perturber_factories,
        blackbox_detector=detector,
        blackbox_scorer=scorer,
        img_batch_size=batch_size,
        verbose=verbose,
    )

    # Compute number of combinations based on provided factories
    num_perturber_combos = 1 if len(perturber_factories) else 0
    for factory in perturber_factories:
        num_perturber_combos *= len(factory)

    assert len(curve) == num_perturber_combos
    assert len(full_scores) == num_perturber_combos
    for scores in full_scores:
        assert len(scores) == len(generator)


def gen_rand_dets(
    im_shape: tuple[int, int],
    n_dets: int,
) -> Sequence[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]:
    """Generate given number of random detections based on image shape.

    :param im_shape: Shape of image the bboxes should fit within.
    :param n_dets: The number of detections to generate.

    :return: The sequence of generated random detections.
    """

    def _get_rand_bbox(im_shape: tuple[int, int]) -> AxisAlignedBoundingBox:
        """Generate a random bounding box within given image shape.

        :param im_shape: Shape of image the bbox should fit within.

        :return: The generated bounding box.
        """
        x_vals = list(rng.integers(0, im_shape[0], size=(2,)))
        y_vals = list(rng.integers(0, im_shape[1], size=(2,)))
        return AxisAlignedBoundingBox(
            min_vertex=(min(x_vals), min(y_vals)),
            max_vertex=(max(x_vals), max(y_vals)),
        )

    return [(_get_rand_bbox(im_shape), {"class": rng.random()}) for _ in range(n_dets)]
