import abc
import numpy as np
from typing import Any, Dict, Hashable, List, Sequence, Tuple, Union

from smqtk_core import Plugfigurable
from smqtk_classifier.interfaces.classification_element import CLASSIFICATION_DICT_T
from smqtk_image_io import AxisAlignedBoundingBox

from nrtk.interfaces.perturb_image_factory import PerturbImageFactory


class GenerateBlackboxResponse(Plugfigurable):
    """
    This interface describes the generation of item-response curves and scores for
    object detections/classifications with respect to the given blackbox model after
    input images are perturbed via the blackbox perturber factory. Scoring of
    these results is computed with the given blackbox scorer.
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
    ) -> Union[Tuple[np.ndarray, Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]], Dict[str, Any]],
               Tuple[np.ndarray, CLASSIFICATION_DICT_T, Dict[str, Any]]]:
        """ Get the ``idx``th image and groundtruth pair. """
