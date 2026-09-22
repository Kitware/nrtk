"""This module contains wrappers for NRTK perturbers for object detection."""

from __future__ import annotations

__all__ = ["MAITEMultiobjectTrackingAugmentation"]

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from maite.protocols import AugmentationMetadata
from maite.protocols.multiobject_tracking import (
    Augmentation,
    DatumMetadataType,
    InputType,
    SingleFrameObjectTrackingTarget,
    TargetType,
)
from maite.protocols.multiobject_tracking import VideoFrame as MAITEVideoFrameProtocol
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.interfaces import PerturbVideo, VideoFrame
from nrtk.interop._maite.metadata import NRTKDatumMetadata
from nrtk.interop._maite.metadata._nrtk_datum_metadata import _forward_md_keys
from nrtk.utils._array import to_numpy

MULTIOBJECT_TRACKING_BATCH_T = tuple[Sequence[InputType], Sequence[TargetType], Sequence[DatumMetadataType]]


@dataclass
class MAITESingleFrameObjectTrackingTarget:
    boxes: np.ndarray[Any, Any]
    labels: np.ndarray[Any, Any]
    scores: np.ndarray[Any, Any]
    track_ids: np.ndarray[Any, Any]


@dataclass
class MAITEVideoFrame:
    pixels: np.ndarray[Any, Any]
    time_s: float
    pts: int
    frame_index: int


@dataclass
class MAITEMultiobjectTrackingTarget:
    frame_tracks: Sequence[MAITESingleFrameObjectTrackingTarget]


class MAITEMultiobjectTrackingAugmentation(Augmentation):
    """Implementation of MAITE Multiobject Tracking Augmentation for NRTK perturbers.

    Implementation of MAITE Augmentation for NRTK perturbers
    operating on a MAITE-protocol compliant Multiobject Tracking dataset.

    Attributes:
        augment : PerturbVideo
            The NRTK PerturbVideo implementation to apply as a MAITE Augmentation.
        metadata: AugmentationMetadata
            Metadata for this augmentation.
    """

    def __init__(self, *, augment: PerturbVideo, augment_id: str) -> None:
        """Initialize augmentation wrapper.

        Args:
            augment:
                PerturbVideo implementation to perform.
            augment_id:
                Metadata ID for this augmentation.
        """
        self.augment = augment
        self.metadata: AugmentationMetadata = AugmentationMetadata(id=augment_id)

    @staticmethod
    def _maite_to_nrtk_frame(
        *,
        frame: MAITEVideoFrameProtocol,
        single_frame_target: SingleFrameObjectTrackingTarget,
        datum_params: dict[str, Any] | None = None,
    ) -> VideoFrame:
        # maite makes time_s optional for containers without a usable time base, but
        # VideoFrame requires a timestamp, so an absent one cannot be carried across.
        if frame.time_s is None:
            raise ValueError("MAITE video frame has no time_s; NRTK's VideoFrame requires a timestamp.")

        # copy=True here only preserves what np.array did; nothing writes to these.
        frame_bboxes = [
            AxisAlignedBoundingBox(min_vertex=bbox[0:2], max_vertex=bbox[2:4])
            for bbox in to_numpy(single_frame_target.boxes, copy=True)
        ]
        frame_labels = [
            {label: score}
            for label, score in zip(
                to_numpy(single_frame_target.labels, copy=True),
                to_numpy(single_frame_target.scores, copy=True),
                strict=True,
            )
        ]

        return VideoFrame(
            image=np.transpose(to_numpy(frame.pixels), (1, 2, 0)).copy(),
            timestamp=frame.time_s,
            boxes=zip(frame_bboxes, frame_labels, strict=True),
            additional_params={
                **(datum_params if datum_params else {}),
                "pts": frame.pts,
                "frame_index": frame.frame_index,
                "track_ids": to_numpy(single_frame_target.track_ids, copy=True),
            },
        )

    @staticmethod
    def _nrtk_to_maite_frame(
        frame: VideoFrame,
    ) -> tuple[MAITEVideoFrame, MAITESingleFrameObjectTrackingTarget]:
        maite_frame = MAITEVideoFrame(
            pixels=np.transpose(frame.image, (2, 0, 1)),
            time_s=frame.timestamp,
            pts=frame.additional_params["pts"],
            frame_index=frame.additional_params["frame_index"],
        )

        try:
            aug_bboxes, aug_score_dicts = zip(*frame.boxes, strict=True)
            aug_bboxes_arr = np.vstack([np.hstack((bbox.min_vertex, bbox.max_vertex)) for bbox in aug_bboxes])
            aug_labels, aug_scores = zip(
                *[
                    # get (label, score) pair for highest score
                    max(score_dict.items(), key=lambda x: x[1])
                    for score_dict in aug_score_dicts
                ],
                strict=True,
            )
        except ValueError:  # No boxes provided
            aug_bboxes_arr = np.empty((0, 4))
            aug_labels = np.empty((0,))
            aug_scores = np.empty((0, 0))  # We do not know the number of classes, so this is only semi-compliant

        maite_single_frame_target = MAITESingleFrameObjectTrackingTarget(
            boxes=aug_bboxes_arr,
            labels=np.asarray(aug_labels),
            scores=np.asarray(aug_scores),
            track_ids=frame.additional_params["track_ids"],
        )

        return maite_frame, maite_single_frame_target

    @override
    def __call__(
        self,
        batch: MULTIOBJECT_TRACKING_BATCH_T,
    ) -> MULTIOBJECT_TRACKING_BATCH_T:
        """Return a batch of augmented data and metadata.

        WARNING: Track IDs are currently passed through unchanged, even if bboxes are modified or removed.
        """
        video_streams, all_anns, metadata = batch

        # Iterate over (parallel) elements in batch
        aug_video_streams = []
        aug_targets = []
        aug_metadata = []

        for video_stream, anns, md in zip(video_streams, all_anns, metadata, strict=True):
            # Convert video stream to NRTK semantics
            nrtk_video_stream = [
                MAITEMultiobjectTrackingAugmentation._maite_to_nrtk_frame(
                    frame=frame,
                    single_frame_target=single_frame_target,
                    datum_params=dict(md),
                )
                for frame, single_frame_target in zip(video_stream, anns.frame_tracks, strict=True)
            ]

            aug_data = self.augment(frames=iter(nrtk_video_stream))

            # Convert back to MAITE semantics
            aug_video_stream = []
            aug_multiframe_target = []
            for frame in aug_data:
                aug_frame, aug_target = MAITEMultiobjectTrackingAugmentation._nrtk_to_maite_frame(frame)
                aug_video_stream.append(aug_frame)
                aug_multiframe_target.append(aug_target)
            aug_video_streams.append(aug_video_stream)
            aug_targets.append(MAITEMultiobjectTrackingTarget(frame_tracks=aug_multiframe_target))

            # Add NRTK perturber config to metadata
            perturber_configs = []
            if "nrtk_perturber_config" in md:
                md_configs = md["nrtk_perturber_config"]
                if TYPE_CHECKING and not isinstance(md_configs, Iterable):  # pragma: no cover
                    raise RuntimeError("Expected iterable perturber config")
                perturber_configs = list(md_configs)
            perturber_configs.append(self.augment.get_config())
            aug_md = NRTKDatumMetadata(
                id=md["id"],
                nrtk_perturber_config=perturber_configs,
            )

            aug_metadata.append(_forward_md_keys(md=md, aug_md=aug_md, forwarded_keys=["id", "nrtk_perturber_config"]))

        return aug_video_streams, aug_targets, aug_metadata
