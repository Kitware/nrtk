import contextlib
from typing import Any
from typing import Dict
from typing import Hashable
from typing import Sequence
from typing import Tuple

# 3rd party imports
from pycocotools.coco import COCO  # type: ignore
from pycocotools.cocoeval import COCOeval  # type: ignore
from smqtk_image_io import AxisAlignedBoundingBox

from nrtk.interfaces.score_detections import ScoreDetections


class COCOScorer(ScoreDetections):
    """
    An implementation of the ``ScoreDetection`` interface that conforms to the
    COCO data formatting and metrics. An instance of this class reads in the
    path to the ground truth data and specifies a particular statistic index.
    Finally, the call to the scorer method returns a set of float metric values
    for the specified statistic index.
    """
    def __init__(self, gt_path: str, stat_index: int = 0) -> None:
        self.gt_path = gt_path
        self.stat_index = stat_index

        with contextlib.redirect_stdout(None):
            self.coco_gt = COCO(gt_path)

        self.cat_ids = {v["name"]: k for k, v in self.coco_gt.cats.items()}

    def score(
        self,
        actual: Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, Any]]]],
        predicted: Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]]
    ) -> Sequence[float]:
        """
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
                raise ValueError("Actual bounding boxes must have detections and can't be empty.")

        for act_dets, pred_dets in zip(actual, predicted):
            image_id = act_dets[0][1]["image_id"]
            batch_ids.append(image_id)

            for bbox, cls_info in act_dets:
                box = [
                    bbox.min_vertex[0],
                    bbox.min_vertex[1],
                    bbox.max_vertex[0]-bbox.min_vertex[0],
                    bbox.max_vertex[1]-bbox.min_vertex[1]
                ]
                box = list(map(float, box))

                entry = {
                    # every image has at least one bounding box
                    "image_id": cls_info["image_id"],
                    "category_id": cls_info["category_id"],
                    "bbox": box
                }

                actual_entries.append(entry)

            for bbox, scores in pred_dets:
                max_score_id = max(scores, key=scores.get)  # type: ignore

                box = [
                    bbox.min_vertex[0],
                    bbox.min_vertex[1],
                    bbox.max_vertex[0]-bbox.min_vertex[0],
                    bbox.max_vertex[1]-bbox.min_vertex[1]
                ]
                box = list(map(float, box))

                entry = {
                    # every image has at least one bounding box
                    "image_id": image_id,
                    "category_id": self.cat_ids[max_score_id],
                    "score": scores[max_score_id],
                    "bbox": box
                }

                predicted_entries.append(entry)

        if len(predicted_entries) == 0:
            return [0]*len(actual)

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

    def get_config(self) -> Dict[str, Any]:
        return {
            "gt_path": self.gt_path,
            "stat_index": self.stat_index
        }
