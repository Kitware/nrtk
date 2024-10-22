from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.interfaces.score_detections import ScoreDetections


class RandomScorer(ScoreDetections):
    """An implementation of the ``ScoreDetection`` interface that serves as a simple test for reproduciblity.

    An instance of this class acts as a functor to generate scores for a specific metric based on a given set
    of ground truth and predicted detections.

    This class, in particular, implements a random scorer that returns random float values.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(self.seed)

    @override
    def score(
        self,
        actual: Sequence[Sequence[tuple[AxisAlignedBoundingBox, dict[Hashable, Any]]]],
        predicted: Sequence[Sequence[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]],
    ) -> Sequence[float]:
        """Return sequence of random float values equal to the length of the ground truth input."""
        if len(actual) != len(predicted):
            raise ValueError("Size mismatch between actual and predicted data")
        for actual_det in actual:
            if len(actual_det) < 1:
                raise ValueError(
                    "Actual bounding boxes must have detections and can't be empty.",
                )

        return [self._rng.random() for actual_det in actual]

    @override
    def get_config(self) -> dict[str, Any]:
        return {"seed": self.seed}
