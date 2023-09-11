from typing import Any, Sequence, Tuple, Dict, Hashable, ContextManager
from contextlib import nullcontext as does_not_raise
import pytest

from smqtk_image_io.bbox import AxisAlignedBoundingBox
from smqtk_core.configuration import configuration_test_helper

from nrtk.impls.score_detections.random_scorer import RandomScorer

from .test_scorer_utils import _class_map, scorer_assertions


class TestRandomScorer:

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
        scorer = RandomScorer()

        with expectation:
            # Test scorer interface directly
            scorer_assertions(scorer=scorer.score, actual=actual, predicted=predicted)
            # Test callable
            scorer_assertions(scorer=scorer, actual=actual, predicted=predicted)

    @pytest.mark.parametrize("random_seed, actual, predicted", [
        (10, dummy_actual, dummy_predicted),
        (234, dummy_actual, dummy_predicted),
        (5678, dummy_actual, dummy_predicted)
    ])
    def test_reproducibility(self,
                             random_seed: int,
                             actual: Sequence[Sequence[Tuple[AxisAlignedBoundingBox,
                                                             Dict[Hashable, Any]]]],
                             predicted: Sequence[Sequence[Tuple[AxisAlignedBoundingBox,
                                                                Dict[Hashable, float]]]]) -> None:
        """
        Test if the returned random sequence is the same for a particular random_seed
        when used on two different object method calls of the scorer.
        """
        scorer_1 = RandomScorer(random_seed)
        scores_sequence_1 = scorer_1(actual=actual, predicted=predicted)

        scorer_2 = RandomScorer(random_seed)
        scores_sequence_2 = scorer_2(actual=actual, predicted=predicted)

        assert scores_sequence_1 == scores_sequence_2

    @pytest.mark.parametrize("random_seed", [10, 234, 5678])
    def test_config(self, random_seed: int) -> None:
        """
        Test configuration stability
        """
        scorer = RandomScorer(random_seed)
        for i in configuration_test_helper(scorer):
            assert i.rng == random_seed
