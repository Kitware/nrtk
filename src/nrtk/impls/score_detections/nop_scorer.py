"""
This module provides the `NOPScorer` class, an implementation of the `ScoreDetections`
interface that serves as a "no operation" (NOP) scorer. The `NOPScorer` class is useful
in testing or baseline scenarios where all input detections receive a score of zero.

Classes:
    NOPScorer: An example implementation of `ScoreDetections` that outputs zero scores
    for each provided ground truth input.

Dependencies:
    - smqtk_image_io for handling bounding box objects.
    - nrtk.interfaces for the `ScoreDetections` interface.

Usage:
    Instantiate `NOPScorer` and use it to generate zero scores for any set of
    ground truth and predicted detections:

    scorer = NOPScorer()
    scores = scorer.score(actual_detections, predicted_detections)

Example:
    nop_scorer = NOPScorer()
    zero_scores = nop_scorer.score(actual_detections, predicted_detections)
"""

from collections.abc import Hashable, Sequence
from typing import Any

from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.interfaces.score_detections import ScoreDetections


class NOPScorer(ScoreDetections):
    """Example implementation of the ``ScoreDetection`` interface.

    An instance of this class acts as a functor to generate scores for a specific metric
    based on a given set of ground truth and predicted detections.

    This class, in particular, serves as a pass-through "no operation" (NOP)
    scorer.
    """

    @override
    def score(
        self,
        actual: Sequence[Sequence[tuple[AxisAlignedBoundingBox, dict[Hashable, Any]]]],
        predicted: Sequence[Sequence[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]],
    ) -> Sequence[float]:
        """Return sequence of zeros equal to the length of the ground truth input."""
        if len(actual) != len(predicted):
            raise ValueError("Size mismatch between actual and predicted data")
        for actual_det in actual:
            if len(actual_det) < 1:
                raise ValueError(
                    "Actual bounding boxes must have detections and can't be empty.",
                )

        return [0 for actual_det in actual]

    def get_config(self) -> dict[str, Any]:
        """
        Returns the current configuration of the NOPScorer instance.

        Returns:
            dict[str, Any]: Configuration dictionary with current settings.
        """
        return {}
