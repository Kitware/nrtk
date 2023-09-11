from typing import Any, Sequence, Tuple, Dict, Hashable, ContextManager
from contextlib import nullcontext as does_not_raise
import pytest

from smqtk_image_io.bbox import AxisAlignedBoundingBox
from smqtk_core.configuration import configuration_test_helper

from nrtk.impls.score_detections.nop_scorer import NOPScorer

from .test_scorer_utils import _class_map, scorer_assertions


class TestNOPScorer:

    dummy_actual = [[(AxisAlignedBoundingBox([1, 1], [2, 2]), {"category": "dummy_class_1"})]]
    dummy_pred_box = AxisAlignedBoundingBox([1, 1], [2, 2])
    dummy_pred_class = _class_map(classes=("dummy_class_1", "dummy_class_2"), scores=[0.9, 0.1])
    dummy_predicted = [[(dummy_pred_box, dummy_pred_class)]]

    dummy_actual_test_len_mismatch = [[(AxisAlignedBoundingBox([1, 1], [2, 2]),
                                        {"category": "dummy_class_1"})],
                                      [(AxisAlignedBoundingBox([2, 2], [3, 3]),
                                        {"category": "dummy_class_2"})]]

    dummy_empty = [[]]  # type: ignore

    @pytest.mark.parametrize("actual, predicted, expectation", [
        (dummy_actual, dummy_predicted, does_not_raise()),
        (dummy_actual_test_len_mismatch, dummy_predicted,
         pytest.raises(ValueError,
                       match=r"Size mismatch between actual and predicted data")),
        (dummy_empty, dummy_empty,
         pytest.raises(ValueError,
                       match=r"Actual bounding boxes must have detections and can't be empty."))
    ])
    def test_basic_assertions_and_exceptions(self,
                                             actual: Sequence[Sequence[Tuple[AxisAlignedBoundingBox,
                                                                             Dict[Hashable, Any]]]],
                                             predicted: Sequence[Sequence[Tuple[AxisAlignedBoundingBox,
                                                                                Dict[Hashable, float]]]],
                                             expectation: ContextManager) -> None:
        """
        Test basic scorer assertions and exceptions using the helper function
        from the utils file.
        """
        scorer = NOPScorer()

        with expectation:
            # Test scorer interface directly
            scorer_assertions(scorer=scorer.score, actual=actual, predicted=predicted)
            # Test callable
            scorer_assertions(scorer=scorer, actual=actual, predicted=predicted)

    def test_config(self) -> None:
        """
        Test configuration stability (check if config is an empty dict)
        """
        scorer = NOPScorer()
        configuration_test_helper(scorer)
