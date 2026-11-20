import io
import logging
from collections.abc import Hashable, Iterable
from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from datamaite import (
    DatasetMetadata,
    ImageObjectDetectionSample,
    ObjectDetectionAnnotation,
    ObjectDetectionDataset,
    Task,
)
from datamaite.taxonomy import CategoryEntry, Taxonomy
from PIL import Image
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.entrypoints._nrtk_eo._nrtk_eo_image_perturber import nrtk_eo_image_perturber
from tests.fakes import FakeImagePerturber, FakePerturbFactory
from tests.utils import random_image

DETECTIONS: dict[int | str, list[tuple[float, float, float, float]]] = {
    0: [(1.0, 2.0, 3.0, 4.0)],
    1: [(0.0, 0.0, 10.0, 5.0), (5.0, 5.0, 2.0, 2.0), (8.0, 1.0, 4.0, 7.0)],
    "1": [(20.0, 30.0, 6.0, 6.0), (2.5, 3.5, 1.5, 2.5)],
}
LOGGER = "nrtk.entrypoints._nrtk_eo._nrtk_eo_image_perturber"
SHIFT = (10.0, 20.0)


class _BoxDroppingImagePerturber(FakeImagePerturber):
    """Drops the last box of any image that has more than one."""

    @override
    def perturb(
        self,
        *,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        **_: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        if boxes is None:
            return np.copy(image), None
        kept = list(boxes)
        return np.copy(image), kept[:-1] if len(kept) > 1 else kept


class _ShiftingImagePerturber(FakeImagePerturber):
    """Translates every box by SHIFT so the xyxy->xywh inverse is observable."""

    @override
    def perturb(
        self,
        *,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        shift: tuple[float, float] = SHIFT,
        **_: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        if boxes is None:
            return np.copy(image), None
        shifted = [
            (
                AxisAlignedBoundingBox(
                    min_vertex=np.asarray(bbox.min_vertex) + shift,
                    max_vertex=np.asarray(bbox.max_vertex) + shift,
                ),
                deepcopy(meta),
            )
            for bbox, meta in boxes
        ]
        return np.copy(image), shifted


class _GSDImagePerturber(_ShiftingImagePerturber):
    """Shift boxes horizontally by the supplied GSD."""

    @override
    def perturb(
        self,
        *,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        img_gsd: float = 0.0,
        **_: Any,
    ) -> tuple[
        np.ndarray[Any, Any],
        Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None,
    ]:
        return super().perturb(
            image=image,
            boxes=boxes,
            shift=(img_gsd, 0.0),
        )


def _png_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def _make_od_dataset(with_metadata: bool = False) -> ObjectDetectionDataset:
    """Build a small datamaite object-detection dataset backed by in-memory PNGs."""
    samples = [
        ImageObjectDetectionSample(
            image_id=image_id,
            file_name=f"images/{idx}.png",
            image_bytes=_png_bytes(random_image(size=(64, 64, 3), seed=idx)),
            width=64,
            height=64,
            detections=tuple(
                ObjectDetectionAnnotation(
                    bbox=bbox,
                    category_id=1,
                    category_name="cat",
                    area=bbox[2] * bbox[3] if with_metadata else None,
                    segmentation=[list(bbox)] if with_metadata else None,
                )
                for bbox in boxes
            ),
        )
        for idx, (image_id, boxes) in enumerate(DETECTIONS.items())
    ]
    taxonomy = Taxonomy(entries=(CategoryEntry(source_id=1, name="cat"),))
    return ObjectDetectionDataset(
        samples=tuple(samples),
        dataset_metadata=DatasetMetadata(taxonomy=taxonomy),
        dataset_id="test_dataset",
    )


@pytest.mark.maite
@pytest.mark.tools
class TestNRTKEOImagePerturber:
    def test_nrtk_eo_image_perturber(self) -> None:
        """Perturber yields one datamaite OD dataset per parameter combination."""
        dataset = _make_od_dataset()
        perturber_factory = FakePerturbFactory(
            perturber=FakeImagePerturber,
            theta_key="param1",
            theta_values=[1, 3],
        )
        img_dirs = ["_param1-1", "_param1-3"]
        augmented_datasets = list(nrtk_eo_image_perturber(dataset=dataset, perturber_factory=perturber_factory))

        assert len(augmented_datasets) == len(img_dirs)
        for perturber_params, aug_dataset in augmented_datasets:
            assert perturber_params in img_dirs
            assert isinstance(aug_dataset, ObjectDetectionDataset)
            assert aug_dataset.task == Task.OD
            assert len(aug_dataset) == len(DETECTIONS)
            assert aug_dataset.num_detections == sum(len(boxes) for boxes in DETECTIONS.values())
            assert all(sample.image_bytes is not None for sample in aug_dataset.samples)
            assert aug_dataset.dataset_metadata.taxonomy is not None
            assert aug_dataset.dataset_metadata.taxonomy == dataset.dataset_metadata.taxonomy

            for idx, sample in enumerate(aug_dataset.samples):
                assert sample.file_name == f"{idx}.png"
                assert sample.image_bytes is not None
                with Image.open(io.BytesIO(sample.image_bytes)) as image:
                    assert image.format == "PNG"
                    assert image.size == (sample.width, sample.height) == (64, 64)
                correct_dets = DETECTIONS[sample.image_id]
                for sample_det, correct_det in zip(sample.detections, correct_dets, strict=True):
                    assert sample_det.bbox == correct_det
                    assert sample_det.category_id == 1
                    assert sample_det.category_name == "cat"

    @pytest.mark.parametrize("with_metadata", [True, False])
    def test_area_and_segmentation_are_dropped(self, with_metadata: bool, caplog: pytest.LogCaptureFixture) -> None:
        """area/segmentation are nulled, and the warning fires only when the source actually had them."""
        dataset = _make_od_dataset(with_metadata=with_metadata)
        perturber_factory = FakePerturbFactory(
            perturber=FakeImagePerturber,
            theta_key="param1",
            theta_values=[1, 3],
        )
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            _, aug_dataset = list(nrtk_eo_image_perturber(dataset=dataset, perturber_factory=perturber_factory))[0]

        for s in aug_dataset.samples:
            for det in s.detections:
                assert det.area is None
                assert det.segmentation is None
        assert any("Area and Segmentation" in rec.message for rec in caplog.records) is with_metadata

    def test_dropped_boxes_are_rebuilt_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """When a perturber drops boxes, survivors are rebuilt from the target and the user is warned once."""
        dataset = _make_od_dataset()
        perturber_factory = FakePerturbFactory(
            perturber=_BoxDroppingImagePerturber,
            theta_key="param1",
            theta_values=[1],
        )
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            _, aug_dataset = list(nrtk_eo_image_perturber(dataset=dataset, perturber_factory=perturber_factory))[0]

        for sample in aug_dataset.samples:
            source = DETECTIONS[sample.image_id]
            expected = source[:-1] if len(source) > 1 else source
            assert [det.bbox for det in sample.detections] == expected
            assert all(det.category_id == 1 and det.category_name == "cat" for det in sample.detections)
            assert all(det.area is None and det.segmentation is None for det in sample.detections)

        messages = [rec.message for rec in caplog.records if "box count changed" in rec.message]
        assert len(messages) == 1
        assert "2 image(s)" in messages[0]
        assert "2 box(es)" in messages[0]

    def test_shifted_boxes_convert_xyxy_to_xywh(self) -> None:
        """A translating perturber moves the origin and preserves width/height."""
        dataset = _make_od_dataset()
        perturber_factory = FakePerturbFactory(
            perturber=_ShiftingImagePerturber,
            theta_key="param1",
            theta_values=[1, 3],
        )
        _, aug_dataset = list(nrtk_eo_image_perturber(dataset=dataset, perturber_factory=perturber_factory))[0]
        for sample in aug_dataset.samples:
            expected = [(x + SHIFT[0], y + SHIFT[1], w, h) for x, y, w, h in DETECTIONS[sample.image_id]]
            assert [det.bbox for det in sample.detections] == expected

    def test_metadata_reaches_perturber(self) -> None:
        """Datum metadata is merged per image id and forwarded to the perturber as kwargs."""
        dataset = _make_od_dataset()
        metadata = {idx: {"shift": (10.0 * (int(idx) + 1), 5.0 * (int(idx) + 1))} for idx in DETECTIONS}

        _, aug_dataset = next(
            iter(
                nrtk_eo_image_perturber(
                    dataset=dataset,
                    perturber_factory=FakePerturbFactory(
                        perturber=_ShiftingImagePerturber,
                        theta_key="param1",
                        theta_values=[1],
                    ),
                    metadata=metadata,
                ),
            ),
        )

        for sample in aug_dataset.samples:
            dx, dy = metadata[sample.image_id]["shift"]
            expected = [(x + dx, y + dy, w, h) for x, y, w, h in DETECTIONS[sample.image_id]]
            assert [det.bbox for det in sample.detections] == expected

    @pytest.mark.parametrize(
        ("image_ids", "metadata", "expected_gsds"),
        [
            ([1], {1: {"img_gsd": 0.25}}, [0.25]),
            (["frame-a"], {"frame-a": {"img_gsd": 0.25}}, [0.25]),
            (["1"], {"1": {"img_gsd": 0.25}}, [0.25]),
            ([1], {"1": {"img_gsd": 0.25}}, [0.0]),
            (["1"], {1: {"img_gsd": 0.25}}, [0.0]),
            (
                [1, "1"],
                {1: {"img_gsd": 0.25}, "1": {"img_gsd": 0.50}},
                [0.25, 0.50],
            ),
            ([1, "1"], {"1": {"img_gsd": 0.50}}, [0.0, 0.50]),
            (
                [1],
                {1: {"img_gsd": 0.25}, "1": {"img_gsd": 0.50}},
                [0.25],
            ),
        ],
    )
    def test_metadata_matches_id_value_and_type(
        self,
        image_ids: list[int | str],
        metadata: dict[int | str, dict[str, Any]],
        expected_gsds: list[float],
    ) -> None:
        """Apply only metadata matching the image ID value and type."""
        source = _make_od_dataset()
        dataset = replace(
            source,
            samples=tuple(
                replace(
                    source.samples[0],
                    image_id=image_id,
                    file_name=f"images/{idx}.png",
                )
                for idx, image_id in enumerate(image_ids)
            ),
        )

        _, augmented = next(
            iter(
                nrtk_eo_image_perturber(
                    dataset=dataset,
                    perturber_factory=FakePerturbFactory(
                        perturber=_GSDImagePerturber,
                        theta_key="param1",
                        theta_values=[1],
                    ),
                    metadata=metadata,
                ),
            ),
        )

        assert len(augmented.samples) == len(image_ids)
        for original, sample, expected_gsd in zip(
            dataset.samples,
            augmented.samples,
            expected_gsds,
            strict=True,
        ):
            assert sample.image_id == original.image_id
            assert type(sample.image_id) is type(original.image_id)
            expected_boxes = [
                (x + expected_gsd, y, width, height)
                for x, y, width, height in (detection.bbox for detection in original.detections)
            ]
            assert [det.bbox for det in sample.detections] == expected_boxes
