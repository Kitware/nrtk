"""
This module defines the `RandomScorer` class, which implements the `ScoreDetections` interface
to generate random scores for object detection tasks. The `RandomScorer` class serves as a
test tool to verify reproducibility by returning random float scores based on a given seed
value. This can be useful in testing scenarios where deterministic random values are needed.

Classes:
    RandomScorer: An implementation of `ScoreDetections` that generates random scores
    for input detection data.

Dependencies:
    - numpy for generating random float values.
    - smqtk_image_io for bounding box handling.
    - nrtk.interfaces for the `ScoreDetections` interface.

Usage:
    Instantiate `RandomScorer` with an optional seed for reproducibility:

    scorer = RandomScorer(seed=42)
    scores = scorer.score(actual_detections, predicted_detections)

Example:
    random_scorer = RandomScorer(seed=123)
    random_scores = random_scorer.score(actual_detections, predicted_detections)
"""

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
        """
        Initializes the `RandomScorer` with an optional random seed.

        Args:
            seed (int | None, optional): An optional seed to ensure reproducibility of random
                scores. Defaults to None.

        """
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

    def get_config(self) -> dict[str, Any]:
        """
        Returns the current configuration of the RandomScorer instance.

        Returns:
            dict[str, Any]: Configuration dictionary with current settings.
        """
        return {"seed": self.seed}
