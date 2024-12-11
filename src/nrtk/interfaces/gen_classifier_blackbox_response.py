"""
This module defines the `GenerateClassifierBlackboxResponse` interface, which provides
functionality for generating item-response curves and scores for image classification tasks
in a blackbox context. The interface allows images to be perturbed and then classified,
with scores computed based on the classification results.

Classes:
    GenerateClassifierBlackboxResponse: An interface for generating item-response curves
    and scores for perturbed image classifications using a specified classifier and scoring mechanism.

Dependencies:
    - numpy for handling image data and computations.
    - smqtk_classifier for image classification.
    - tqdm for progress tracking.
    - smqtk_core for plugin configuration.
    - nrtk.interfaces for blackbox response generation, perturbation, and scoring interfaces.

Usage:
    To create an instance of `GenerateClassifierBlackboxResponse`, implement the required
    abstract methods, such as `__getitem__`, and use the `generate` method to produce
    item-response curves and scores.

Example:
    class CustomClassifierResponse(GenerateClassifierBlackboxResponse):
        def __getitem__(self, idx):
            # Implementation of data retrieval
            pass

    generator = CustomClassifierResponse()
    response_curve, scores = generator.generate(
        blackbox_perturber_factories=[factory1, factory2],
        blackbox_classifier=classifier,
        blackbox_scorer=scorer,
        img_batch_size=32,
        verbose=True
    )
"""

import abc
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

import numpy as np
from smqtk_classifier import ClassifyImage
from smqtk_classifier.interfaces.classification_element import CLASSIFICATION_DICT_T
from tqdm import tqdm
from typing_extensions import override

from nrtk.interfaces.gen_blackbox_response import (
    GenerateBlackboxResponse,
    gen_perturber_combinations,
)
from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from nrtk.interfaces.score_classifications import ScoreClassifications


class GenerateClassifierBlackboxResponse(GenerateBlackboxResponse):
    """This interface describes generation of item-response curves and scores for image classification w.r.t. blackbox.

    This interface describes the generation of item-response curves and scores for
    image classifications with respect to the given blackbox classifer after
    input images are perturbed via the blackbox perturber factory. Scoring of
    these detections is computed with the given blackbox scorer.
    """

    @override
    @abc.abstractmethod
    def __getitem__(
        self,
        idx: int,
    ) -> tuple[np.ndarray, CLASSIFICATION_DICT_T, dict[str, Any]]:
        """Get the ``idx``th image and ground_truth pair."""

    def generate(  # noqa C901:
        self,
        blackbox_perturber_factories: Sequence[PerturbImageFactory],
        blackbox_classifier: ClassifyImage,
        blackbox_scorer: ScoreClassifications,
        img_batch_size: int,
        verbose: bool = False,
    ) -> tuple[Sequence[tuple[dict[str, Any], float]], Sequence[Sequence[float]]]:
        """Generate item-response curves for given parameters.

        :param blackbox_perturber_factories: Sequence of factories to perturb stimuli.
        :param blackbox_classifier: Classifier to generate calssifications for perturbed stimuli.
        :param blackbox_scorer: Scorer to score classifications.
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

                batch_predicted = blackbox_classifier.classify_images(batch_images)

                scores = blackbox_scorer(
                    actual=batch_gt,
                    predicted=list(batch_predicted),  # Interface requires list not iterator
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
        blackbox_classifier: ClassifyImage,
        blackbox_scorer: ScoreClassifications,
        img_batch_size: int,
        verbose: bool = False,
    ) -> tuple[Sequence[tuple[dict[str, Any], float]], Sequence[Sequence[float]]]:
        """Alias for :meth: ``.GenerateClassifierBlackboxResponse.generate``."""
        return self.generate(
            blackbox_perturber_factories=blackbox_perturber_factories,
            blackbox_classifier=blackbox_classifier,
            blackbox_scorer=blackbox_scorer,
            img_batch_size=img_batch_size,
            verbose=verbose,
        )
