"""A wrapper class for a YOLO model to simplify its usage with input batches and object detection targets."""

__all__ = ["MaiteYOLODetector"]

from collections.abc import Sequence

from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.utils._exceptions import NRTKXAITKHelperImportError
from nrtk.utils._import_guard import import_guard

from .dataset import YOLODetectionTarget

maite_available: bool = import_guard(
    module_name="maite",
    exception=NRTKXAITKHelperImportError,
    submodules=["protocols.object_detection"],
    objects=["Model", "InputType"],
) and import_guard(
    module_name="maite",
    exception=NRTKXAITKHelperImportError,
    submodules=["protocols"],
    objects=["ModelMetadata"],
)
torch_available: bool = import_guard(module_name="torch", exception=NRTKXAITKHelperImportError, fake_spec=True)
ultralytics_available: bool = import_guard(
    module_name="ultralytics",
    exception=NRTKXAITKHelperImportError,
    submodules=["models"],
)
nrtk_xaitk_helpers_available: bool = maite_available and torch_available and ultralytics_available
import torch  # noqa: E402
import ultralytics.models  # noqa: E402
from maite.protocols import ModelMetadata  # noqa: E402
from maite.protocols.object_detection import InputType, Model  # noqa: E402


class MaiteYOLODetector(Model):
    """A wrapper class for a YOLO model to simplify its usage with input batches and object detection targets.

    This class takes a YOLO model instance, processes input image batches, and converts predictions into
    `YOLODetectionTarget` instances.

    Attributes:
        _model (ultralytics.models.yolo.model.YOLO): The YOLO model instance used for predictions.

    Methods:
        __call__(batch):
            Processes a batch of images through the YOLO model and returns the predictions as
            `YOLODetectionTarget` instances.
    """

    def __init__(self, model: ultralytics.models.yolo.model.YOLO) -> None:
        """Initializes the MaiteYOLODetector with a YOLO model instance.

        Args:
            model (ultralytics.models.yolo.model.YOLO): The YOLO model to use for predictions.
        """
        self._model = model
        # Dummy model metadata type to pass type checking
        self.metadata: ModelMetadata = ModelMetadata(id="0")

    def __call__(self, batch: Sequence[InputType]) -> Sequence[YOLODetectionTarget]:
        """Processes a batch of images using the YOLO model and converts the predictions to `YOLODetectionTarget`s.

        Args:
            batch (Sequence[ArrayLike]): A batch of images in (c, h, w) format (channel-first).

        Returns:
            Sequence[YOLODetectionTarget]: A list of YOLODetectionTarget instances containing the predictions for each
            image in the batch.
        """
        imgs = [img.transpose(axes=(1, 2, 0)) if img.shape[-1] != 3 else img for img in batch]  # pyright: ignore [reportAttributeAccessIssue]

        with torch.no_grad():
            yolo_predictions = self._model(imgs, stream=True, verbose=False)  # Run inference

        # yolo_predictions = self._model(batch, verbose=False)

        return [
            YOLODetectionTarget(
                boxes=p.boxes.xyxy.cpu(),  # Bounding boxes in (x_min, y_min, x_max, y_max) format
                labels=p.boxes.cls.cpu(),  # Class indices for the detected objects
                scores=p.boxes.conf.cpu(),  # Confidence scores for the detections
            )
            for p in yolo_predictions
        ]

    def detect_objects(
        self,
        imgs: Sequence[InputType],
    ) -> Sequence[tuple[AxisAlignedBoundingBox, dict[int, float]]]:
        """Detect objects in an iterable of images and return bounding boxes with scores.

        Args:
            imgs (Sequence[InputType]): A sequence of images in (h, w, c) format.

        Returns:
            Iterable[Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]: Detections for each image.
        """
        preds = []
        for img in imgs:
            detections = self([img])
            for det in detections:
                results = []
                for bbox, label, score in zip(det.boxes, det.labels, det.scores, strict=False):
                    score_dict = dict.fromkeys(range(10), 0.0)  # Ensure all class labels exist
                    score_dict[int(label.item())] = float(score.item())
                    bbox = AxisAlignedBoundingBox(
                        min_vertex=(bbox[0].item(), bbox[1].item()),
                        max_vertex=(bbox[2].item(), bbox[3].item()),
                    )
                    results.append(
                        (bbox, score_dict),
                    )
                preds.append(results)
        return preds
