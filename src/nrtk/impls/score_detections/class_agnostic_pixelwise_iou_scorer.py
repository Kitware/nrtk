# standard library imports
from typing import Any, Dict, Hashable, Sequence, Tuple

# 3rd party imports
import numpy as np
from smqtk_image_io import AxisAlignedBoundingBox

# local imports
from nrtk.interfaces.score_detections import ScoreDetections


class ClassAgnosticPixelwiseIoUScorer(ScoreDetections):
    """Implementation of `ScoreDetection` interface that computes the Pixelwise IoU scores in a Class-Agnostic manner.

    The call to the scorer method returns a sequence of float values containing the Pixelwise IoU
    scores for the specified ground truth and predictions inputs.
    """

    def score(
        self,
        actual: Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, Any]]]],
        predicted: Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]],
    ) -> Sequence[float]:
        """Computes pixelwise IoU scores and returns sequence of float values equal to the length of the input data."""
        if len(actual) != len(predicted):
            raise ValueError("Size mismatch between actual and predicted data")
        for actual_det in actual:
            if len(actual_det) < 1:
                raise ValueError("Actual bounding boxes must have detections and can't be empty.")

        ious = list()

        for act, pred in zip(actual, predicted):
            width, height = 1, 1
            for act_bbox, _ in act:
                width = max(width, act_bbox.max_vertex[0])
                height = max(width, act_bbox.max_vertex[1])

            for pred_bbox, _ in pred:
                width = max(width, pred_bbox.max_vertex[0])
                height = max(width, pred_bbox.max_vertex[1])

            width = int(width) + 1
            height = int(height) + 1

            actual_mask = np.zeros((height, width), dtype=bool)
            predicted_mask = np.zeros((height, width), dtype=bool)

            for act_bbox, _ in act:
                x_1, y_1 = act_bbox.min_vertex
                x_2, y_2 = act_bbox.max_vertex
                actual_mask[int(y_1) : int(y_2), int(x_1) : int(x_2)] = 1  # noqa: E203

            for pred_bbox, _ in pred:
                x_1, y_1 = pred_bbox.min_vertex
                x_2, y_2 = pred_bbox.max_vertex
                # Black formatting keeps moving the noqa comment down a line, which causes flake8 error
                # fmt: off
                predicted_mask[int(y_1) : int(y_2), int(x_1) : int(x_2)] = (  # noqa: E203
                    1
                )
                # fmt: on

            intersection = np.logical_and(actual_mask, predicted_mask)
            union = np.logical_or(actual_mask, predicted_mask)

            ious.append(np.sum(intersection) / np.sum(union))

        return ious

    def get_config(self) -> Dict[str, Any]:
        return {}
