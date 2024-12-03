"""
This module provides the `COCOScorer` class, an implementation of the `ScoreDetections`
interface that calculates object detection scores conforming to the COCO (Common Objects in
Context) dataset format and evaluation metrics. This class is specifically tailored to use COCO
evaluation metrics by formatting ground truth and predicted data accordingly.

Classes:
    COCOScorer: Scores object detection results based on COCO evaluation metrics, allowing
    users to specify ground truth data and a particular statistic index to retrieve a sequence
    of metric scores.

Dependencies:
    - pycocotools for COCO data handling and evaluation.
    - smqtk_image_io for bounding box handling.
    - nrtk.interfaces for the `ScoreDetections` interface.
    - contextlib for suppressing unwanted output during COCO initialization.

Usage:
    Instantiate `COCOScorer` with the path to the COCO-formatted ground truth data and specify
    the statistic index to be used for scoring. Call `score` with actual and predicted bounding
    boxes to obtain a sequence of float scores.

Example:
    scorer = COCOScorer(gt_path="path/to/ground_truth.json", stat_index=0)
    scores = scorer.score(actual_detections, predicted_detections)
"""

import contextlib
from collections.abc import Hashable, Sequence
from typing import Any

# 3rd party imports
from pycocotools.coco import COCO  # type: ignore
from pycocotools.cocoeval import COCOeval  # type: ignore
from smqtk_image_io import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.interfaces.score_detections import ScoreDetections


class COCOScorer(ScoreDetections):
    """An implementation of the ``ScoreDetection`` interface that conforms to the COCO data formatting and metrics.

    An instance of this class reads in the
    path to the ground truth data and specifies a particular statistic index.
    Finally, the call to the scorer method returns a set of float metric values
    for the specified statistic index.
    """

    def __init__(self, gt_path: str, stat_index: int = 0) -> None:
        """
        Initializes the `COCOScorer` with a path to the ground truth data and a statistic index.

        Args:
            gt_path (str): Path to the COCO-formatted ground truth JSON file.
            stat_index (int): Index of the statistic to retrieve from the COCO evaluation. Defaults to 0.
        """
        self.gt_path = gt_path
        self.stat_index = stat_index

        with contextlib.redirect_stdout(None):
            self.coco_gt = COCO(gt_path)

        self.cat_ids = {v["name"]: k for k, v in self.coco_gt.cats.items()}

    @override
    def score(  # noqa: C901
        self,
        actual: Sequence[Sequence[tuple[AxisAlignedBoundingBox, dict[Hashable, Any]]]],
        predicted: Sequence[Sequence[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]],
    ) -> Sequence[float]:
        """Computes scores for a particular statistic index.

        Computes scores for a particular statistic index and returns sequences of float values
        equal to the length of the input data.
        """
        actual_entries = list()
        predicted_entries = list()
        batch_ids = list()

        if len(actual) != len(predicted):
            raise ValueError("Size mismatch between actual and predicted data")
        for actual_det in actual:
            if len(actual_det) < 1:
                raise ValueError(
                    "Actual bounding boxes must have detections and can't be empty.",
                )

        for act_dets, pred_dets in zip(actual, predicted):
            image_id = act_dets[0][1]["image_id"]
            batch_ids.append(image_id)

            for bbox, cls_info in act_dets:
                box = [
                    bbox.min_vertex[0],
                    bbox.min_vertex[1],
                    bbox.max_vertex[0] - bbox.min_vertex[0],
                    bbox.max_vertex[1] - bbox.min_vertex[1],
                ]
                box = list(map(float, box))

                entry = {
                    # every image has at least one bounding box
                    "image_id": cls_info["image_id"],
                    "category_id": cls_info["category_id"],
                    "bbox": box,
                }

                actual_entries.append(entry)

            for bbox, scores in pred_dets:
                max_score_id = max(scores, key=scores.get)  # type: ignore

                box = [
                    bbox.min_vertex[0],
                    bbox.min_vertex[1],
                    bbox.max_vertex[0] - bbox.min_vertex[0],
                    bbox.max_vertex[1] - bbox.min_vertex[1],
                ]
                box = list(map(float, box))

                entry = {
                    # every image has at least one bounding box
                    "image_id": image_id,
                    "category_id": self.cat_ids[max_score_id],
                    "score": scores[max_score_id],
                    "bbox": box,
                }

                predicted_entries.append(entry)

        if len(predicted_entries) == 0:
            return [0] * len(actual)

        with contextlib.redirect_stdout(None):
            actual_coco_dt = self.coco_gt.loadRes(actual_entries)
            predicted_coco_dt = self.coco_gt.loadRes(predicted_entries)

        coco_eval = COCOeval(actual_coco_dt, predicted_coco_dt, "bbox")

        final_scores = list()
        for img_id in list(set(batch_ids)):
            coco_eval.params.imgIds = img_id

            with contextlib.redirect_stdout(None):
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()

            final_scores.append(coco_eval.stats[self.stat_index])

        return final_scores

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration dictionary for the `COCOScorer` instance.

        Returns:
            dict[str, Any]: Configuration dictionary containing the ground truth path and statistic index.
        """
        return {"gt_path": self.gt_path, "stat_index": self.stat_index}
