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

    @override
    def get_config(self) -> dict[str, Any]:
        return {}
