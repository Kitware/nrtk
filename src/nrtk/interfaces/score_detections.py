"""
This module defines the `ScoreDetections` interface, which abstracts the process of scoring
object detection outputs based on specified metrics. Implementations of this interface are
expected to compute scores by comparing actual and predicted bounding box detections, with
corresponding class labels and confidence scores.

Classes:
    ScoreDetections: Interface for scoring detection outputs against ground truth data.

Dependencies:
    - smqtk_core for configurable interface capabilities.
    - smqtk_image_io for handling Axis-Aligned Bounding Boxes.

Usage:
    To create a custom detection scoring class, inherit from `ScoreDetections` and
    implement the `score` method, ensuring validation and consistency of input data
    as required.

Example:
    class CustomScorer(ScoreDetections):
        def score(self, actual, predicted):
            # Implementation of custom scoring logic
            pass

    scorer = CustomScorer()
    scores = scorer(actual_detections, predicted_detections)
"""

import abc
from collections.abc import Hashable, Sequence
from typing import Any

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
        actual: Sequence[Sequence[tuple[AxisAlignedBoundingBox, dict[Hashable, Any]]]],
        predicted: Sequence[Sequence[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]],
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
        actual: Sequence[Sequence[tuple[AxisAlignedBoundingBox, dict[Hashable, Any]]]],
        predicted: Sequence[Sequence[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]],
    ) -> Sequence[float]:
        """Alias for :meth:`.ScoreDetection.score`."""
        return self.score(actual, predicted)
