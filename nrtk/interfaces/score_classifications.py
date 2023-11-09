import abc
from typing import Sequence

from smqtk_core import Plugfigurable
from smqtk_classifier.interfaces.classification_element import CLASSIFICATION_DICT_T


class ScoreClassifications(Plugfigurable):
    """
    Interface abstracting the behavior of taking the actual and predicted
    classifications and computing the corresponding metric scores.

    Implementations should verify the validity of the input data.
    """

    @abc.abstractmethod
    def score(
        self,
        actual: Sequence[CLASSIFICATION_DICT_T],
        predicted: Sequence[CLASSIFICATION_DICT_T]
    ) -> Sequence[float]:
        """
        Generate a sequence of scores corresponding to a specific metric.

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
        predicted: Sequence[CLASSIFICATION_DICT_T]
    ) -> Sequence[float]:
        """
        Alias for :meth:`.ScoreClassifications.score`.
        """
        return self.score(actual, predicted)
