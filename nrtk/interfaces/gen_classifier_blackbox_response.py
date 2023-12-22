import abc
import numpy as np
from contextlib import nullcontext
from tqdm import tqdm
from typing import Any, Dict, List, Sequence, Tuple

from smqtk_classifier import ClassifyImage
from smqtk_classifier.interfaces.classification_element import CLASSIFICATION_DICT_T

from nrtk.interfaces.gen_blackbox_response import GenerateBlackboxResponse
from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from nrtk.interfaces.score_classifications import ScoreClassifications


class GenerateClassifierBlackboxResponse(GenerateBlackboxResponse):
    """
    This interface describes the generation of item-response curves and scores for
    image classifications with respect to the given blackbox classifer after
    input images are perturbed via the blackbox perturber factory. Scoring of
    these detections is computed with the given blackbox scorer.
    """
    @abc.abstractmethod
    def __getitem__(
        self,
        idx: int
    ) -> Tuple[np.ndarray, CLASSIFICATION_DICT_T, Dict[str, Any]]:
        """ Get the ``idx``th image and groundtruth pair. """

    def generate(
        self,
        blackbox_perturber_factories: Sequence[PerturbImageFactory],
        blackbox_classifier:          ClassifyImage,
        blackbox_scorer:              ScoreClassifications,
        img_batch_size:               int,
        verbose:                      bool = False
    ) -> Tuple[Sequence[Tuple[Dict[str, Any], float]], Sequence[Sequence[float]]]:
        """
        Generate item-response curves for given parameters.

        :param blackbox_perturber_factories: Sequence of factories to perturb stimuli.
        :param blackbox_classifier: Classifier to generate calssifications for perturbed stimuli.
        :param blackbox_scorer: Scorer to score classifications.
        :param img_batch_size: The number of images to predict and score upon at once.
        :param verbose: Increases the verbosity of progress updates.

        :return: Item-response curve
        :return: Scores for each input stimuli
        """
        curve: List[Tuple[Dict[str, Any], float]] = list()
        full: List[Sequence[float]] = list()

        def process(perturbers: Sequence[PerturbImage]) -> None:
            """
            Generate item-response curve and individual stimuli scores for
            this set of perturbers.

            :param perturbers: Set of perturbers to perturb image stimuli.
            """
            image_scores: List[float] = list()

            # Generate batch of images and GT detections so we can predict
            # and score in batches
            for i in range(0, len(self), img_batch_size):
                batch_images = list()
                batch_gt = list()
                for j in range(i, min(i + img_batch_size, len(self))):
                    image, actual, extra = self[j]
                    perturbed = image.copy()

                    for perturber in perturbers:
                        perturbed = perturber(perturbed, extra)

                    batch_images.append(perturbed)
                    batch_gt.append(actual)

                batch_predicted = blackbox_classifier.classify_images(batch_images)

                scores = blackbox_scorer(
                    actual=batch_gt,
                    predicted=[b for b in batch_predicted]  # Interface requires list not iterator
                )
                image_scores.extend(scores)

            # Get theta values for each perturber in set as independent variables of item-response curve
            x = {
                factory.theta_key: getattr(perturbers[idx], factory.theta_key)
                for idx, factory in enumerate(blackbox_perturber_factories)
            }

            # Add item-response values (summary and individual) to results
            curve.append(
                (x, float(np.mean(image_scores)))
            )
            full.append(image_scores)

        # Generate results for each combination of perturbers
        # Note: order of factories is preserved when applying pertubations
        pert_combos = GenerateBlackboxResponse._gen_perturber_combinations(
            factories=blackbox_perturber_factories
        )
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
        blackbox_classifier:          ClassifyImage,
        blackbox_scorer:              ScoreClassifications,
        img_batch_size:               int,
        verbose:                      bool = False
    ) -> Tuple[Sequence[Tuple[Dict[str, Any], float]], Sequence[Sequence[float]]]:
        """
        Alias for :meth: ``.GenerateClassifierBlackboxResponse.generate``.
        """
        return self.generate(
            blackbox_perturber_factories=blackbox_perturber_factories,
            blackbox_classifier=blackbox_classifier,
            blackbox_scorer=blackbox_scorer,
            img_batch_size=img_batch_size,
            verbose=verbose
        )
