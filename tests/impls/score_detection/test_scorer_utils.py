from typing import Callable, Dict, Hashable, Sequence, Tuple
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


def scorer_assertions(
    scorer: Callable[[Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]],
                      Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]]], Sequence[float]],
    actual: Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]],
    predicted: Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]]
) -> None:
    """
    Basic scorer assertions:
    1) Test to make sure the scorer implementation does not
    modify the input data.
    2) Test to make sure the inputs are 2D sequences.
    3) Test to make sure the scorer output is the same length
    as the ground truth data.
    """
    actual_copy = copy.deepcopy(actual)
    predicted_copy = copy.deepcopy(predicted)

    scores_sequence = scorer(actual, predicted)

    assert actual == actual_copy
    assert predicted == predicted_copy

    assert (isinstance(actual, Sequence) and len(actual) > 0 and isinstance(actual[0], Sequence))
    assert (isinstance(predicted, Sequence) and len(predicted) > 0 and isinstance(predicted[0], Sequence))

    assert len(scores_sequence) == len(actual)
