from typing import Any
from typing import Dict
from typing import Hashable
from typing import List
from typing import Tuple
from typing import Optional

import random

from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.interfaces.score_detection import ScoreDetection


class RandomScorer(ScoreDetection):
    """
    An implementation of the ``ScoreDetection`` interface that serves as a
    simple test for reproduciblity. An instance of this class acts as a
    functor to generate scores for a specific metric based on a given set
    of ground truth and predicted detections.

    This class, in particular, implements a random scorer that returns random
    float values.
    """
    def __init__(self, rng: Optional[int] = None):
        self.rng = rng
        random.seed(self.rng)

    def score(
        self,
        actual: List[List[Tuple[AxisAlignedBoundingBox, str]]],
        predicted: List[List[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]],
    ) -> List[float]:
        """
        Return list of random float values equal to the length of the ground truth input.
        """
        if len(actual) != len(predicted):
            raise ValueError("Size mismatch between actual and predicted data")
        for actual_det in actual:
            if len(actual_det) < 1:
                raise ValueError("Actual bounding boxes must have detections and can't be empty.")

        return [random.random() for actual_det in actual]

    def get_config(self) -> Dict[str, Any]:
        return {
            "rng": self.rng
        }
