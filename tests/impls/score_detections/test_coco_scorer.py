import json
import tempfile
from contextlib import nullcontext as does_not_raise
from typing import Any, ContextManager, Dict, Hashable, Sequence, Tuple

import numpy as np
import pytest
from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.impls.score_detections.coco_scorer import COCOScorer

from .test_scorer_utils import _class_map, scorer_assertions


class TestCOCOScorer:
    dummy_actual = [[(AxisAlignedBoundingBox([1, 1], [2, 2]), {"category_id": 1, "image_id": 1})]]
    dummy_pred_box = AxisAlignedBoundingBox([1, 1], [2, 2])
    dummy_pred_class = _class_map(classes=("dummy_class_1", "dummy_class_2"), scores=[0.9, 0.1])
    dummy_predicted = [[(dummy_pred_box, dummy_pred_class)]]

    dummy_actual_test_len_mismatch = [
        [(AxisAlignedBoundingBox([1, 1], [2, 2]), {"category_id": 1, "image_id": 1})],
        [(AxisAlignedBoundingBox([2, 2], [3, 3]), {"category_id": 2, "image_id": 1})],
    ]

    dummy_empty = [[]]  # type: ignore

    annotation_data = {
        "annotations": [
            {
                "area": 447.64974501609544,
                "bbox": [1.0, 1.0, 2.0, 2.0],
                "category_id": 1,
                "id": 1,
                "image_id": 1,
                "iscrowd": 0,
            }
        ],
        "images": [{"id": 1, "file_name": "dummy_img.tif", "width": 128.0, "height": 128.0}],
        "categories": [
            {"id": 1, "name": "dummy_class_1"},
            {"id": 2, "name": "dummy_class_2"},
        ],
    }

    @pytest.mark.parametrize(
        ("actual", "predicted", "annotation_data", "expectation"),
        [
            (dummy_actual, dummy_predicted, annotation_data, does_not_raise()),
            (dummy_actual, dummy_empty, annotation_data, does_not_raise()),
            (
                dummy_actual_test_len_mismatch,
                dummy_predicted,
                annotation_data,
                pytest.raises(ValueError, match=r"Size mismatch between actual and predicted data"),
            ),
            (
                dummy_empty,
                dummy_empty,
                annotation_data,
                pytest.raises(
                    ValueError,
                    match=r"Actual bounding boxes must have detections and can't be empty.",
                ),
            ),
        ],
    )
    def test_basic_assertions_and_exceptions(
        self,
        actual: Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, Any]]]],
        predicted: Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]],
        annotation_data: Dict,
        expectation: ContextManager,
    ) -> None:
        """Test basic scorer assertions and exceptions using the helper function from the utils file."""
        tmp_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".json")
        json.dump(annotation_data, tmp_file)
        tmp_file.flush()
        json_filename = tmp_file.name
        scorer = COCOScorer(gt_path=json_filename, stat_index=0)

        with expectation:
            # Test scorer interface directly
            scorer_assertions(scorer=scorer.score, actual=actual, predicted=predicted)
            # Test callable
            scorer_assertions(scorer=scorer, actual=actual, predicted=predicted)

    @pytest.mark.parametrize(
        ("actual", "predicted", "annotation_data", "stat_index", "expectation"),
        [
            (dummy_actual, dummy_predicted, annotation_data, 0, does_not_raise()),
            (dummy_actual, dummy_predicted, annotation_data, -1, does_not_raise()),
            (
                dummy_actual,
                dummy_predicted,
                annotation_data,
                12,
                pytest.raises(
                    IndexError,
                    match=r"index 12 is out of bounds for axis 0 with size 12",
                ),
            ),
        ],
    )
    def test_gt_and_predictions_validity(
        self,
        actual: Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, Any]]]],
        predicted: Sequence[Sequence[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]],
        annotation_data: Dict,
        stat_index: int,
        expectation: ContextManager,
    ) -> None:
        """Test validity of the ground truth and predictions."""
        tmp_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".json")
        json.dump(annotation_data, tmp_file)
        tmp_file.flush()
        json_filename = tmp_file.name

        with open(json_filename, mode="r", encoding="utf-8") as file:
            ann_data = file.read()

        ann_json = json.loads(ann_data)

        with expectation:
            # Load annotation data from json and check if the values match the input GT
            act = actual[0][0]
            bbox, cls_info = act
            assert cls_info["image_id"] == ann_json["annotations"][0]["image_id"]
            assert cls_info["category_id"] == ann_json["annotations"][0]["category_id"]
            assert np.array_equal(
                np.concatenate([bbox.min_vertex, bbox.max_vertex]),
                np.array(ann_json["annotations"][0]["bbox"]),
            )

            # Check if GT loaded by COCOScorer obj is non-empty
            scorer = COCOScorer(gt_path=json_filename, stat_index=stat_index)
            assert scorer.coco_gt

            # Check if predictions are valid
            pred = predicted[0][0]
            pred_bbox, scores = pred
            idx = max(scores, key=scores.get)  # type: ignore

            cats = ann_json["categories"]

            assert any(scorer.cat_ids[idx] == c["id"] for c in cats)
            assert all(np.concatenate([pred_bbox.min_vertex, pred_bbox.max_vertex]) >= np.array([0.0, 0.0, 0.0, 0.0]))

            scorer.score(actual=actual, predicted=predicted)

    @pytest.mark.parametrize("annotation_data", [annotation_data])
    def test_config(self, annotation_data: Dict) -> None:
        """Test configuration stability."""
        tmp_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".json")
        json.dump(annotation_data, tmp_file)
        tmp_file.flush()
        json_filename = tmp_file.name
        stat_index = 0
        scorer = COCOScorer(gt_path=json_filename, stat_index=stat_index)
        for i in configuration_test_helper(scorer):
            assert i.gt_path == json_filename
            assert i.stat_index == stat_index
