"""
This module defines the `ScoreClassifications` interface, which provides an abstract
method for scoring classification outputs based on a chosen metric. This interface is
intended for implementations that compute metric scores by comparing actual and predicted
classification results.

Classes:
    ScoreClassifications: Interface for scoring classification outputs against ground truth
    data, with an expectation of producing metric-based scores.

Dependencies:
    - smqtk_core for providing a configurable plugin interface.
    - smqtk_classifier for handling classification dictionary typing.

Usage:
    To create a custom classification scoring class, inherit from `ScoreClassifications`
    and implement the `score` method, ensuring the input data is validated as required.

Example:
    class CustomClassificationScorer(ScoreClassifications):
        def score(self, actual, predicted):
            # Implement custom scoring logic
            pass

    scorer = CustomClassificationScorer()
    scores = scorer(actual_classifications, predicted_classifications)
"""

import abc
from collections.abc import Sequence

from smqtk_classifier.interfaces.classification_element import CLASSIFICATION_DICT_T
from smqtk_core import Plugfigurable


class ScoreClassifications(Plugfigurable):
    """Interface abstracting the behavior of taking classifications and computing corresponding metric scores.

    Interface abstracting the behavior of taking the actual and predicted classifications and
    computing the corresponding metric scores.

    Implementations should verify the validity of the input data.
    """

    @abc.abstractmethod
    def score(
        self,
        actual: Sequence[CLASSIFICATION_DICT_T],
        predicted: Sequence[CLASSIFICATION_DICT_T],
    ) -> Sequence[float]:
        """Generate a sequence of scores corresponding to a specific metric.

        :param actual:
            Ground truth classifications.
        :param predicted:
            Output classifications from a classifier with class-wise confidence
            scores.

        :return:
            Metric score values as a float-type sequence with the length matching
            the number of samples in the ground truth input.
        """

    def __call__(
        self,
        actual: Sequence[CLASSIFICATION_DICT_T],
        predicted: Sequence[CLASSIFICATION_DICT_T],
    ) -> Sequence[float]:
        """Alias for :meth:`.ScoreClassifications.score`."""
        return self.score(actual, predicted)
