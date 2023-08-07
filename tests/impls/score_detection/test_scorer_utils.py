from typing import List, Dict, Tuple
from typing import Sequence, Hashable, Callable
import copy

from smqtk_image_io.bbox import AxisAlignedBoundingBox


def _class_map(classes: Sequence, scores: Sequence) -> Dict[Hashable, float]:
    """
    Mapping function that returns the class-wise scores dict.
    """
    d = {}
    for i, c in enumerate(classes):
        d[c] = scores[i]

    return d


def scorer_assertions(scorer: Callable[[List[List[Tuple[AxisAlignedBoundingBox, str]]],
                                        List[List[Tuple[AxisAlignedBoundingBox,
                                                        Dict[Hashable, float]]]]], List[float]],
                      actual: List[List[Tuple[AxisAlignedBoundingBox, str]]],
                      predicted: List[List[Tuple[AxisAlignedBoundingBox,
                                                 Dict[Hashable, float]]]]) -> None:
    """
    Basic scorer assertions:
    1) Test to make sure the scorer implementation does not
    modify the input data.
    2) Test to make sure the inputs are 2D lists.
    3) Test to make sure the scorer output is the same length
    as the ground truth data.
    """
    actual_copy = copy.deepcopy(actual)
    predicted_copy = copy.deepcopy(predicted)

    scores_list = scorer(actual, predicted)

    assert actual == actual_copy
    assert predicted == predicted_copy

    assert (isinstance(actual, list) and len(actual) > 0 and isinstance(actual[0], list))
    assert (isinstance(predicted, list) and len(predicted) > 0 and isinstance(predicted[0], list))

    assert len(scores_list) == len(actual)
