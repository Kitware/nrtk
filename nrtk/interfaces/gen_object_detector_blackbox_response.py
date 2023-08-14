import abc
import numpy as np
from contextlib import nullcontext
from tqdm import tqdm
from typing import Dict, Hashable, List, Sequence, Tuple

from smqtk_core import Plugfigurable
from smqtk_detection import DetectImageObjects
from smqtk_image_io import AxisAlignedBoundingBox

from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from nrtk.interfaces.score_detection import ScoreDetection


class GenerateObjectDetectorBlackboxResponse(Plugfigurable):
    """
    This interface describes the generation of item-response curves and scores for
    object detections with respect to the given blackbox object detector after
    input images are perturbed via the blackbox perturber factory. Scoring of
    these detections is computed with the given blackbox scorer.
    """

    @staticmethod
    def _gen_perturber_combinations(factories: Sequence[PerturbImageFactory]) -> List[List[int]]:
        """
        Generates list of perturber combinations, including one perturber from each factory.

        :param factories: Sequence of factories from which to generate combinations.

        :return: List of perturber combinations. For each combination, the value, ``x``,
                at index ``y`` corresponds to the ``x``th perturber of the ``y``th factory.
        """
        def _gen(factory_id: int, factory_sizes: List[int]) -> List[List[int]]:
            """
            Recursive method to build up list of lists of perturber combinations

            :param factory_id: Index of the factory to add to the combinations.
            :param factory_sizes: List of the number of perturbers per factory.

            :return: List of combinations.
            """
            # List with every perturber index for this factory as its own list
            if factory_id == len(factory_sizes) - 1:
                return [[i] for i in range(factory_sizes[factory_id])]

            # Generate downstream combinations
            combos = _gen(factory_id=factory_id + 1, factory_sizes=factory_sizes)

            # Add each perturber for this factory to the beginning of all downstream combinations
            out = list()
            for perturber_id in range(factory_sizes[factory_id]):
                for combo in combos:
                    out.append([perturber_id] + combo)
            return out

        factory_sizes = [len(factory) for factory in factories]
        return _gen(factory_id=0, factory_sizes=factory_sizes)

    @abc.abstractmethod
    def __len__(self) -> int:
        """ Number of images for which responses will be generated. """

    @abc.abstractmethod
    def __getitem__(
        self,
        idx: int
    ) -> Tuple[np.ndarray, Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]]:
        """ Get the ``idx``th image and groundtruth pair. """

    def generate(
        self,
        blackbox_perturber_factories: Sequence[PerturbImageFactory],
        blackbox_detector:            DetectImageObjects,
        blackbox_scorer:              ScoreDetection,
        img_batch_size:               int,
        verbose:                      bool = False
    ) -> Tuple[Sequence[Tuple[Tuple[float, ...], float]], Sequence[Sequence[float]]]:
        """
        Generate item-response curves for given parameters.

        :param blackbox_perturber_factories: Sequence of factories to perturb stimuli.
        :param blackbox_detector: Detector to generate detections for perturbed stimuli.
        :param blackbox_scorer: Scorer to score detections.
        :param img_batch_size: The number of images to predict and score upon at once.
        :param verbose: Increases the verbosity of progress updates.

        :return: Item-response curve
        :return: Scores for each input stimuli
        """
        curve = list()
        full = list()

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
                    image, actual = self[j]
                    perturbed = image.copy()

                    for perturber in perturbers:
                        perturbed = perturber(perturbed)

                    batch_images.append(perturbed)
                    batch_gt.append(actual)

                batch_predicted = blackbox_detector(batch_images)

                scores = blackbox_scorer(
                    actual=batch_gt,
                    predicted=[list(b) for b in batch_predicted]  # Interface requires list not iterable
                )
                image_scores.extend(scores)

            # Get theta values for each perturber in set as independent variables of item-response curve
            x = [
                getattr(perturbers[idx], factory.theta_key)
                for idx, factory in enumerate(blackbox_perturber_factories)
            ]

            # Add item-response values (summary and individual) to results
            curve.append(
                (tuple(x), float(np.mean(image_scores)))
            )
            full.append(image_scores)

        # Generate results for each combination of perturbers
        # Note: order of factories is preserved when applying pertubations
        pert_combos = GenerateObjectDetectorBlackboxResponse._gen_perturber_combinations(
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
        blackbox_detector:            DetectImageObjects,
        blackbox_scorer:              ScoreDetection,
        img_batch_size:               int,
        verbose:                      bool = False
    ) -> Tuple[Sequence[Tuple[Tuple[float, ...], float]], Sequence[Sequence[float]]]:
        """
        Alias for :meth: ``.GenerateObjectDetectorBlackboxResponse.generate``.
        """
        return self.generate(
            blackbox_perturber_factories=blackbox_perturber_factories,
            blackbox_detector=blackbox_detector,
            blackbox_scorer=blackbox_scorer,
            img_batch_size=img_batch_size,
            verbose=verbose
        )
