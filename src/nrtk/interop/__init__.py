"""Define the nrtk.interop package."""

from nrtk.interop._maite.augmentations.image_classification import MAITEImageClassificationAugmentation
from nrtk.interop._maite.augmentations.object_detection import MAITEObjectDetectionAugmentation

# Override __module__ to reflect the public API path
MAITEImageClassificationAugmentation.__module__ = __name__
MAITEObjectDetectionAugmentation.__module__ = __name__

__all__ = [
    "MAITEImageClassificationAugmentation",
    "MAITEObjectDetectionAugmentation",
]
