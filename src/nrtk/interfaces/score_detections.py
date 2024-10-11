import abc
from typing import Any, Dict, Hashable, Sequence, Tuple

from smqtk_core import Plugfigurable
from smqtk_image_io.bbox import AxisAlignedBoundingBox


class ScoreDetections(Plugfigurable):
    """Interface abstracting the behavior of taking detections and computing the corresponding metric scores.

    Interface abstracting the behavior of taking the actual and predicted detections and
    computing the corresponding metric scores.

    Implementations should verify the validity of the input data.

    Note that current implementations are not required to verify nor correct dimension
    (in)consistency, which may impact scoring.
    """

    @abc.abstractmethod
    def score(
        self,
        actual: Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, Any]]]],
        predicted: Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]],
    ) -> Sequence[float]:
        """Generate a sequence of scores corresponding to a specific metric.

        :param actual:
            Ground truth bbox and class label pairs.
        :param predicted:
            Output detections from a detector with bbox and
            class-wise confidence scores.

        :return:
            Metric score values as a float-type sequence with the length matching
            the number of samples in the ground truth input.
        """

    def __call__(
        self,
        actual: Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, Any]]]],
        predicted: Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]],
    ) -> Sequence[float]:
        """Alias for :meth:`.ScoreDetection.score`."""
        return self.score(actual, predicted)
