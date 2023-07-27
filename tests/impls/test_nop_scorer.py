from typing import Sequence, Dict, Hashable
import copy

from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.impls.score_detection.nop_scorer import NOPScorer


def _class_map(classes: Sequence, scores: Sequence) -> Dict[Hashable, float]:
    """
    Mapping function that returns the class-wise scores dict.
    """
    d = {}
    for i, c in enumerate(classes):
        d[c] = scores[i]

    return d


def test_scorer() -> None:
    """
    Run on dummy values of BBox and labels to test the scorer implementation directly.
    """
    dummy_actual = [[(AxisAlignedBoundingBox([1, 1], [2, 2]), 'dummy_class_1')]]
    dummy_pred_box = AxisAlignedBoundingBox([1, 1], [2, 2])
    dummy_pred_class = _class_map(classes=('dummy_class_1', 'dummy_class_2'), scores=[0.9, 0.1])
    dummy_predicted = [[(dummy_pred_box, dummy_pred_class)]]

    scorer = NOPScorer()
    scores_list = scorer.score(dummy_actual, dummy_predicted)
    assert scores_list == [0]


def test_callable() -> None:
    """
    Test using the ``__call__`` alias.
    """
    dummy_actual = [[(AxisAlignedBoundingBox([1, 1], [2, 2]), 'dummy_class_1')]]
    dummy_pred_box = AxisAlignedBoundingBox([1, 1], [2, 2])
    dummy_pred_class = _class_map(classes=('dummy_class_1', 'dummy_class_2'), scores=[0.9, 0.1])
    dummy_predicted = [[(dummy_pred_box, dummy_pred_class)]]

    scorer = NOPScorer()
    scores_list = scorer(dummy_actual, dummy_predicted)

    assert scores_list == [0]


def test_input_validity() -> None:
    """
    Test to make sure the scorer implementation does not modify the input data
    """
    dummy_actual = [[(AxisAlignedBoundingBox([1, 1], [2, 2]), 'dummy_class_1')]]
    dummy_actual_copy = copy.deepcopy(dummy_actual)

    dummy_pred_box = AxisAlignedBoundingBox([1, 1], [2, 2])
    dummy_pred_class = _class_map(classes=('dummy_class_1', 'dummy_class_2'), scores=[0.9, 0.1])
    dummy_predicted = [[(dummy_pred_box, dummy_pred_class)]]
    dummy_predicted_copy = copy.deepcopy(dummy_predicted)

    scorer = NOPScorer()
    _ = scorer(dummy_actual, dummy_predicted)

    assert dummy_actual == dummy_actual_copy
    assert dummy_predicted == dummy_predicted_copy


def test_input_len() -> None:
    """
    Test to make sure the scorer output is the same length as the ground truth data.
    """
    dummy_actual = [[(AxisAlignedBoundingBox([1, 1], [2, 2]), 'dummy_class_1')]]
    dummy_pred_box = AxisAlignedBoundingBox([1, 1], [2, 2])
    dummy_pred_class = _class_map(classes=('dummy_class_1', 'dummy_class_2'), scores=[0.9, 0.1])
    dummy_predicted = [[(dummy_pred_box, dummy_pred_class)]]

    scorer = NOPScorer()
    scores_list = scorer(dummy_actual, dummy_predicted)

    assert len(scores_list) == len(dummy_actual)


def test_config() -> None:
    """
    Test to check if get_config() returns an empty dict
    """
    scorer = NOPScorer()
    config = scorer.get_config()

    # Config should be empty, but not None
    assert config is not None
    assert not config
