import abc
from typing import Dict
from typing import Hashable
from typing import List
from typing import Tuple

from smqtk_core import Plugfigurable
from smqtk_image_io.bbox import AxisAlignedBoundingBox


class ScoreDetection(Plugfigurable):
    """
    Interface abstracting the behavior of taking the actual and predicted
    detections and computing the corresponding metric scores.

    Implementations should verify the validity of the input data.
    """

    @abc.abstractmethod
    def score(
        self,
        actual: List[List[Tuple[AxisAlignedBoundingBox, str]]],
        predicted: List[List[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]]
    ) -> List[float]:
        """
        Generate a list of scores corresponding to a specific metric.

        :param actual:
            Ground truth bbox and class label pairs.
        :param predicted:
            Output detections from a detector with bbox and
            class-wise confidence scores.

        :return:
            Metric score values as a float-type list with the length matching
            the number of samples in the ground truth input.
        """

    def __call__(
        self,
        actual: List[List[Tuple[AxisAlignedBoundingBox, str]]],
        predicted: List[List[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]]
    ) -> List[float]:
        """
        Alias for :meth:`.ScoreDetection.score`.
        """
        return self.score(actual, predicted)
