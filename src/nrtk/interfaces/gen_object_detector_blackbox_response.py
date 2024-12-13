"""
This module provides the `GenerateObjectDetectorBlackboxResponse` class, an interface
for generating item-response curves and scores for object detection models. The module
also includes functions to handle image perturbations and scoring within a blackbox setting,
using various factories, detectors, and scoring mechanisms.

Classes:
    GenerateObjectDetectorBlackboxResponse: An interface that defines methods to generate
    item-response curves and scores for object detections in response to perturbed images.

Functions:
    gen_perturber_combinations: Generates combinations of perturbers, selecting one from
    each factory.

Dependencies:
    - numpy for handling numerical operations.
    - smqtk_detection for object detection.
    - smqtk_image_io for image bounding box handling.
    - tqdm for progress updates.
    - typing and collections for typing and context management.

Example usage:
    factories = [perturber_factory1, perturber_factory2]
    detector = SomeObjectDetector()
    scorer = SomeScorer()
    generator = GenerateObjectDetectorBlackboxResponse()
    item_response, scores = generator.generate(factories, detector, scorer, img_batch_size=4, verbose=True)
"""

import abc
from collections.abc import Hashable, Sequence
from contextlib import nullcontext
from typing import Any

import numpy as np
from smqtk_detection import DetectImageObjects
from smqtk_image_io import AxisAlignedBoundingBox
from tqdm import tqdm
from typing_extensions import override

from nrtk.interfaces.gen_blackbox_response import (
    GenerateBlackboxResponse,
    gen_perturber_combinations,
)
from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from nrtk.interfaces.score_detections import ScoreDetections


class GenerateObjectDetectorBlackboxResponse(GenerateBlackboxResponse):
    """This interface describes generation of item-response curves and scores for object detection w.r.t. a blackbox.

    This interface describes the generation of item-response curves and scores for
    object detections with respect to the given black-box object detector after
    input images are perturbed via the black-box perturber factory. Scoring of
    these detections is computed with the given black-box scorer.

    Note that dimension transformations are not currently accounted for and may impact scoring.
    """

    @override
    @abc.abstractmethod
    def __getitem__(
        self,
        idx: int,
    ) -> tuple[
        np.ndarray,
        Sequence[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]],
        dict[str, Any],
    ]:
        """Get the image and ground_truth pair at a particular ``idx``."""

    def generate(  # noqa: C901
        self,
        blackbox_perturber_factories: Sequence[PerturbImageFactory],
        blackbox_detector: DetectImageObjects,
        blackbox_scorer: ScoreDetections,
        img_batch_size: int,
        verbose: bool = False,
    ) -> tuple[Sequence[tuple[dict[str, Any], float]], Sequence[Sequence[float]]]:
        """Generate item-response curves for given parameters.

        :param blackbox_perturber_factories: Sequence of factories to perturb stimuli.
        :param blackbox_detector: Detector to generate detections for perturbed stimuli.
        :param blackbox_scorer: Scorer to score detections.
        :param img_batch_size: The number of images to predict and score upon at once.
        :param verbose: Increases the verbosity of progress updates.

        :return: Item-response curve
        :return: Scores for each input stimuli
        """
        curve: list[tuple[dict[str, Any], float]] = list()
        full: list[Sequence[float]] = list()

        def process(perturbers: Sequence[PerturbImage]) -> None:
            """Generate item-response curve and individual stimuli scores for this set of perturbers.

            :param perturbers: Set of perturbers to perturb image stimuli.
            """
            image_scores: list[float] = list()

            # Generate batch of images and GT detections so we can predict
            # and score in batches
            for i in range(0, len(self), img_batch_size):
                batch_images = list()
                batch_gt = list()
                for j in range(i, min(i + img_batch_size, len(self))):
                    image, actual, extra = self[j]
                    perturbed = image.copy()

                    for perturber in perturbers:
                        perturbed, _ = perturber(perturbed, additional_params=extra)

                    batch_images.append(perturbed)
                    batch_gt.append(actual)

                batch_predicted = blackbox_detector(batch_images)

                scores = blackbox_scorer(
                    actual=batch_gt,
                    predicted=[list(b) for b in batch_predicted],  # Interface requires list not iterable
                )
                image_scores.extend(scores)

            # Get theta values for each perturber in set as independent variables of item-response curve
            x = {
                factory.theta_key: getattr(perturbers[idx], factory.theta_key)
                for idx, factory in enumerate(blackbox_perturber_factories)
            }

            # Add item-response values (summary and individual) to results
            curve.append((x, float(np.mean(image_scores))))
            full.append(image_scores)

        # Generate results for each combination of perturbers
        # Note: order of factories is preserved when applying pertubations
        pert_combos = gen_perturber_combinations(factories=blackbox_perturber_factories)
        with tqdm(total=len(pert_combos)) if verbose else nullcontext() as progress_bar:  # type: ignore
            for c in pert_combos:
                perturbers = [factory[p] for factory, p in zip(blackbox_perturber_factories, c)]
                process(perturbers)
                if progress_bar:
                    progress_bar.update(1)

        return curve, full

    def __call__(
        self,
        blackbox_perturber_factories: Sequence[PerturbImageFactory],
        blackbox_detector: DetectImageObjects,
        blackbox_scorer: ScoreDetections,
        img_batch_size: int,
        verbose: bool = False,
    ) -> tuple[Sequence[tuple[dict[str, Any], float]], Sequence[Sequence[float]]]:
        """Alias for :meth: ``.GenerateObjectDetectorBlackboxResponse.generate``."""
        return self.generate(
            blackbox_perturber_factories=blackbox_perturber_factories,
            blackbox_detector=blackbox_detector,
            blackbox_scorer=blackbox_scorer,
            img_batch_size=img_batch_size,
            verbose=verbose,
        )
