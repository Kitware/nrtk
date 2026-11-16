"""nrtk_eo_image_perturber generates augmented image dataset(s) based on a perturber factory configuration."""

__all__ = ["nrtk_eo_image_perturber"]

import io
import itertools
import logging
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from datamaite import (
    ImageObjectDetectionSample,
    ObjectDetectionAnnotation,
    ObjectDetectionDataset,
)
from maite.protocols.object_detection import (
    DatumMetadataType,
)
from PIL import Image

from nrtk.interfaces import PerturbFactory, PerturbImage
from nrtk.interop import MAITEObjectDetectionAugmentation
from nrtk.utils._logging import setup_logging

logger: logging.Logger = setup_logging(name=__name__)


def nrtk_eo_image_perturber(  # noqa: C901 ignore too complex
    *,
    dataset: ObjectDetectionDataset,
    perturber_factory: PerturbFactory[PerturbImage],
    metadata: dict[int | str, dict[str, Any]] | None = None,
) -> Iterable[tuple[str, ObjectDetectionDataset]]:
    """Generate augmented datamaite ``ObjectDetectionDataset``(s) from an OD dataset.

    Args:
        dataset:
            Source datamaite object-detection dataset.
        perturber_factory:
            PerturbFactory implementation.
        metadata:
            Dictionary of per-image metadata to pass to the perturber

    Returns:
        An iterable of ``(perturber_params, ObjectDetectionDataset)`` tuples.
    """
    perturber_factory_config = perturber_factory.get_config()
    if "theta_keys" in perturber_factory_config:  # multivariate factory doesn't follow interface rules
        perturb_factory_keys = perturber_factory_config["theta_keys"]
        thetas = perturber_factory.thetas
    else:
        perturb_factory_keys = [perturber_factory_config["theta_key"]]
        thetas = [perturber_factory.thetas]

    perturber_combinations = [dict(zip(perturb_factory_keys, v, strict=False)) for v in itertools.product(*thetas)]
    logger.info(f"Perturber sweep values: {perturber_combinations}")

    # Iterate through the different perturber factory parameter combinations and
    # build a datamaite dataset of the perturbed images for each.
    logger.info("Starting perturber sweep")
    augmented_datasets: list[ObjectDetectionDataset] = []
    output_perturb_params: list[str] = []
    drops_metadata = any(
        det.area is not None or det.segmentation is not None for sample in dataset.samples for det in sample.detections
    )
    index2label = dataset.index2label()
    for i, (perturber_combo, perturber) in enumerate(zip(perturber_combinations, perturber_factory, strict=False)):
        output_perturb_params.append("".join(f"_{str(k)}-{str(v)}" for k, v in perturber_combo.items()))

        logger.info(f"Starting perturbation for {output_perturb_params[i]}")

        maite_perturber = MAITEObjectDetectionAugmentation(augment=perturber, augment_id=output_perturb_params[i])

        samples: list[ImageObjectDetectionSample] = []
        dropped_dets: dict[int | str, int] = {}
        for idx in range(len(dataset)):
            image, target, meta = dataset[idx]
            src = dataset.samples[idx]
            merged_meta: dict[str, Any] = {**meta}
            if metadata:
                merged_meta.update(metadata.get(meta["id"], {}))
            md = DatumMetadataType(**merged_meta)
            aug_img, aug_det, _ = maite_perturber(
                batch=(
                    [image],
                    [target],
                    [md],
                ),
            )

            pil = Image.fromarray(np.transpose(np.asarray(aug_img)[0], (1, 2, 0)))
            buf = io.BytesIO()
            pil.save(buf, format="PNG")

            # Update the box geometry, area, and segmentation.
            target = aug_det[0]
            boxes, labels = np.asarray(target.boxes), np.asarray(target.labels)
            if len(boxes) != len(src.detections):
                dropped_dets[src.image_id] = len(src.detections) - len(boxes)
                dets = tuple(
                    ObjectDetectionAnnotation(
                        bbox=(float(b[0]), float(b[1]), float(b[2]) - float(b[0]), float(b[3]) - float(b[1])),
                        category_id=int(label) if int(label) != -1 else None,
                        category_name=index2label.get(int(label)),
                    )
                    for b, label in zip(boxes, labels, strict=True)
                )
            else:
                dets = tuple(
                    replace(
                        det,
                        bbox=(float(b[0]), float(b[1]), float(b[2]) - float(b[0]), float(b[3]) - float(b[1])),
                        area=None,  # Dropping since we no longer guarantee area is the same
                        segmentation=None,  # Dropping since we no longer guarantee segmentation is the same
                    )
                    for det, b in zip(src.detections, boxes, strict=True)  # strict=True guards count drift
                )

            file_name = f"{Path(src.file_name).stem}.png" if src.file_name else f"{src.image_id}.png"
            samples.append(
                replace(
                    src,
                    image_bytes=buf.getvalue(),
                    path_or_uri=None,  # perturbed pixels no longer live at the source path
                    file_name=file_name,
                    width=pil.width,
                    height=pil.height,
                    detections=dets,
                ),
            )

        if dropped_dets:
            logger.warning(
                f"{output_perturb_params[i]}: box count changed on {len(dropped_dets)} image(s); "
                f"{sum(dropped_dets.values())} box(es) dropped in total. Annotations were recreated "
                "using only category_id, category_name, and bbox.",
            )

        augmented_datasets.append(
            ObjectDetectionDataset(
                samples=tuple(samples),
                dataset_metadata=dataset.dataset_metadata,  # carry taxonomy/info/licenses through
                dataset_id=output_perturb_params[i],
            ),
        )
    if drops_metadata:
        logger.warning(
            "Area and Segmentation attributes of ObjectDetectionAnnotation are set to None. "
            "Perturbers can modify bounding boxes. Instead of passing possibly incorrect metadata "
            "from source, we elect to drop both values to ensure the new metadata is accurate.",
        )
    return zip(output_perturb_params, augmented_datasets, strict=False)
