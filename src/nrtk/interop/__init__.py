"""Define the nrtk.interop package."""

from nrtk.interop._maite.augmentations.image_classification import MAITEClassificationAugmentation
from nrtk.interop._maite.augmentations.object_detection import MAITEDetectionAugmentation

# Override __module__ to reflect the public API path
MAITEClassificationAugmentation.__module__ = __name__
MAITEDetectionAugmentation.__module__ = __name__

__all__ = [
    "MAITEClassificationAugmentation",
    "MAITEDetectionAugmentation",
]
