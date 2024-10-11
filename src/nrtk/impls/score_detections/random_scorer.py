import random
from typing import Any, Dict, Hashable, Optional, Sequence, Tuple

from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.interfaces.score_detections import ScoreDetections


class RandomScorer(ScoreDetections):
    """An implementation of the ``ScoreDetection`` interface that serves as a simple test for reproduciblity.

    An instance of this class acts as a functor to generate scores for a specific metric based on a given set
    of ground truth and predicted detections.

    This class, in particular, implements a random scorer that returns random float values.
    """

    def __init__(self, rng: Optional[int] = None):
        self.rng = rng
        random.seed(self.rng)

    def score(
        self,
        actual: Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, Any]]]],
        predicted: Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]],
    ) -> Sequence[float]:
        """Return sequence of random float values equal to the length of the ground truth input."""
        if len(actual) != len(predicted):
            raise ValueError("Size mismatch between actual and predicted data")
        for actual_det in actual:
            if len(actual_det) < 1:
                raise ValueError("Actual bounding boxes must have detections and can't be empty.")

        # we include nosemgrep because random.random() is sufficient for our need
        return [random.random() for actual_det in actual]  # nosemgrep

    def get_config(self) -> Dict[str, Any]:
        return {"rng": self.rng}
