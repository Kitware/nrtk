"""
This module defines the `GenerateBlackboxResponse` interface and a utility function
for generating perturber combinations across multiple factories.

Classes:
    GenerateBlackboxResponse: An interface for generating item-response curves and scores
    for object detections/classifications from perturbed images in a blackbox model.

Functions:
    gen_perturber_combinations: Generates combinations of perturbers, selecting one from each factory.

Dependencies:
    - numpy for numerical operations.
    - smqtk_core for configuration and plugin support.
    - smqtk_image_io for image bounding box handling.
    - smqtk_classifier for classification dictionary typing.

Example usage:
    factories = [factory1, factory2]
    perturber_combinations = gen_perturber_combinations(factories)
"""

import abc
from collections.abc import Hashable, Sequence
from typing import Any, Union

import numpy as np
from smqtk_classifier.interfaces.classification_element import CLASSIFICATION_DICT_T
from smqtk_core import Plugfigurable
from smqtk_image_io import AxisAlignedBoundingBox

from nrtk.interfaces.perturb_image_factory import PerturbImageFactory


def gen_perturber_combinations(  # noqa: C901
    factories: Sequence[PerturbImageFactory],
) -> list[list[int]]:
    """Generates list of perturber combinations, including one perturber from each factory.

    :param factories: Sequence of factories from which to generate combinations.

    :return: list of perturber combinations. For each combination, the value, ``x``,
            at index ``y`` corresponds to the ``x``th perturber of the ``y``th factory.
    """
    # make sure at least one factory is passed in
    if len(factories) == 0:
        raise ValueError("No factories passed in, cannot generate perturbations")

    # if factory has length of zero, remove from sequence
    for factory in factories:
        if len(factory) == 0:
            raise ValueError("Factory passed in with invald length of 0")

    def _gen(factory_id: int, factory_sizes: list[int]) -> list[list[int]]:
        """Recursive method to build up list of lists of perturber combinations.

        :param factory_id: Index of the factory to add to the combinations.
        :param factory_sizes: list of the number of perturbers per factory.

        :return: list of combinations.
        """
        # list with every perturber index for this factory as its own list
        if factory_id == len(factory_sizes) - 1:
            return [[i] for i in range(factory_sizes[factory_id])]

        # Generate downstream combinations
        combos = _gen(factory_id=factory_id + 1, factory_sizes=factory_sizes)

        # Add each perturber for this factory to the beginning of all downstream combinations
        return [[perturber_id] + combo for perturber_id in range(factory_sizes[factory_id]) for combo in combos]

    factory_sizes = [len(factory) for factory in factories]
    return _gen(factory_id=0, factory_sizes=factory_sizes)


class GenerateBlackboxResponse(Plugfigurable):
    """This interface describes generation of item-response curves & scores for detection/classification for blackbox.

    This interface describes the generation of item-response curves and scores for
    object detections/classifications with respect to the given blackbox model after
    input images are perturbed via the blackbox perturber factory. Scoring of
    these results is computed with the given blackbox scorer.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Number of images for which responses will be generated."""

    @abc.abstractmethod
    def __getitem__(
        self,
        idx: int,
    ) -> Union[
        tuple[
            np.ndarray,
            Sequence[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]],
            dict[str, Any],
        ],
        tuple[np.ndarray, CLASSIFICATION_DICT_T, dict[str, Any]],
    ]:
        """Get the ``idx``th image and ground_truth pair."""
