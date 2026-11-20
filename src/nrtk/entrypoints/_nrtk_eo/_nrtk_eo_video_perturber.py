"""nrtk_eo_video_perturber generates augmented video dataset(s) based on a perturber factory configuration.

This is an experimental entrypoint (gated behind ``import nrtk.experimental``).
"""

from __future__ import annotations

from typing import Any

__all__ = ["nrtk_eo_video_perturber"]

import itertools
import logging
from collections.abc import Iterable
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from datamaite import BoxAnnotation, BoxTrackDataset, VideoSequence
from maite.protocols.multiobject_tracking import DatumMetadata, InputType, SingleFrameObjectTrackingTarget, TargetType

from nrtk.impls.perturb_video import FramewisePerturber
from nrtk.interfaces import PerturbFactory, PerturbImage, PerturbVideo
from nrtk.interop import MAITEMultiobjectTrackingAugmentation
from nrtk.utils._logging import setup_logging

logger: logging.Logger = setup_logging(name=__name__)


def _boxes_for_frame(
    *,
    source_boxes: list[BoxAnnotation],
    frame_target: SingleFrameObjectTrackingTarget,
    frame_index: int,
    timestamp: float | None,
    uri_by_id: dict[int, str],
    index2label: dict[int, str],
    uuid_by_track: dict[int, str],
) -> list[BoxAnnotation]:
    """Update ``BoxAnnotation`` from xyxy to xywh.

    If boxes were dropped by the perturber, all ``BoxAnnotation`` is reconstructed and ``attributes`` are dropped
    """
    boxes = np.asarray(frame_target.boxes)
    labels = np.asarray(frame_target.labels)
    track_ids = np.asarray(frame_target.track_ids)
    if len(track_ids) != len(boxes):
        track_ids = np.full(len(boxes), -1)
        return [
            BoxAnnotation(
                track_uuid=uuid_by_track.get(int(tid), f"untracked-{frame_index}-{k}"),
                track_id=int(tid),
                category_id=int(label),
                category_uri=uri_by_id.get(int(label), ""),
                category_name=index2label.get(int(label)),
                bbox=(float(b[0]), float(b[1]), float(b[2]) - float(b[0]), float(b[3]) - float(b[1])),
                attributes={},
                frame_index=frame_index,
                timestamp=timestamp,
            )
            for k, (b, label, tid) in enumerate(zip(boxes, labels, track_ids, strict=True))
        ]
    return [
        replace(
            src,
            bbox=(float(b[0]), float(b[1]), float(b[2]) - float(b[0]), float(b[3]) - float(b[1])),
            frame_index=frame_index,
        )
        for src, b in zip(source_boxes, boxes, strict=True)
    ]


def _to_datamaite_sequence(  # noqa: C901 ignore too complex
    *,
    aug_stream: InputType,
    aug_target: TargetType,
    source_seq: VideoSequence,
    staging_dir: Path,
    uri_by_id: dict[int, str],
    index2label: dict[int, str],
) -> VideoSequence:
    """Rebuild one datamaite ``VideoSequence`` from an augmented MAITE MOT item.

    The perturbed frames are encoded into a *single* video file (rather than
    per-frame images) so the reconstructed sequence carries a ``video_path``.
    Video-backed writers (e.g. HMIE) copy that file directly; image-sequence
    writers (e.g. MOTChallenge) decode it back to frames.

    The videos are encoded in ``libx264`` with ``crf`` set to ``0`` and ``pix_fmt``
    set to ``yuv444p``. These options help reduce compression and preserve the
    perturbation with as little loss as possible.
    """
    suffix = Path(source_seq.video_path).suffix if source_seq.video_path else ".mp4"
    video_out = staging_dir / f"{source_seq.video_id}{suffix}"
    video_out.parent.mkdir(parents=True, exist_ok=True)

    rate = (
        Fraction(source_seq.fps).limit_denominator(max_denominator=1000000)
        if source_seq.fps
        else Fraction(numerator=30, denominator=1)
    )
    uuid_by_track = {box.track_id: box.track_uuid for box in source_seq.boxes}

    with av.open(str(video_out), mode="w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.options = {"crf": "0"}  # As little compression as possible
        stream.pix_fmt = "yuv444p"
        by_frame = source_seq.boxes_by_frame()

        boxes: list[BoxAnnotation] = []
        width = source_seq.width if source_seq.width else 0
        height = source_seq.height if source_seq.height else 0
        num_frames = 0
        dropped_boxes = 0
        affected_frames = 0
        frame_ids = (
            range(source_seq.num_frames) if source_seq.num_frames and source_seq.num_frames_exact else sorted(by_frame)
        )
        for i, (frame, frame_target, frame_id) in enumerate(
            zip(aug_stream, aug_target.frame_tracks, frame_ids, strict=True),
        ):
            hwc = np.ascontiguousarray(np.transpose(np.asarray(frame.pixels), (1, 2, 0)))
            if num_frames == 0:
                height, width = hwc.shape[0], hwc.shape[1]
                stream.width = width
                stream.height = height
            video_frame = av.VideoFrame.from_ndarray(hwc[:height, :width], format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
            source_boxes = by_frame.get(frame_id, [])
            frame_boxes = _boxes_for_frame(
                source_boxes=source_boxes,
                frame_target=frame_target,
                frame_index=i,
                timestamp=frame.time_s,
                uri_by_id=uri_by_id,
                index2label=index2label,
                uuid_by_track=uuid_by_track,
            )
            if len(frame_boxes) != len(source_boxes):
                dropped_boxes += len(source_boxes) - len(frame_boxes)
                affected_frames += 1
            boxes.extend(frame_boxes)
            num_frames += 1
        for packet in stream.encode():
            container.mux(packet)

    if affected_frames:
        logger.warning(
            f"video_id={source_seq.video_id}: box count changed on {affected_frames} frame(s), "
            f"{dropped_boxes} box(es) dropped. Those frames' boxes were rebuilt from the perturber "
            "output with track_id=-1 and no attributes, since track identity cannot be recovered after a drop.",
        )

    return VideoSequence(
        video_id=source_seq.video_id,
        video_path=str(video_out),
        fps=source_seq.fps,
        num_frames=num_frames,
        duration=(num_frames / source_seq.fps) if source_seq.fps else None,
        annotation_path=source_seq.annotation_path,
        status=source_seq.status,
        video_meta=dict(source_seq.video_meta),
        metadata=dict(source_seq.metadata),
        boxes=boxes,
        width=width,
        height=height,
        num_frames_exact=True,
    )


def nrtk_eo_video_perturber(  # noqa: C901 ignore too complex
    *,
    dataset: BoxTrackDataset,
    perturber_factory: PerturbFactory[PerturbImage | PerturbVideo],
    staging_dir: str,
    metadata: dict[int | str, dict[str, Any]] | None = None,
) -> Iterable[tuple[str, BoxTrackDataset]]:
    """Generate augmented datamaite ``BoxTrackDataset``(s) from a MOT dataset.

    Args:
        dataset:
            Source datamaite MOT dataset.
        perturber_factory:
            PerturbFactory implementation.
        staging_dir:
            Temporary directory to write videos out to
        metadata:
            Dictionary of per-video metadata to pass to the perturber

    Returns:
        An iterable of ``(perturber_params, BoxTrackDataset)`` tuples.
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

    logger.info(f"Staging perturbed frames under {str(staging_dir)}")

    source_sequences = [seq for seq in dataset.sequences if seq.video_path is not None]

    logger.info("Starting perturber sweep")
    augmented_datasets: list[BoxTrackDataset] = []
    output_perturb_params: list[str] = []

    # Allow non-anotated frames through
    dataset = dataset.with_mot_options(empty_frame_policy="all")
    uri_by_id = {cid: uri for uri, cid in dataset.categories.items()}
    index2label = dataset.index2label()

    for i, (perturber_combo, perturber) in enumerate(zip(perturber_combinations, perturber_factory, strict=False)):
        output_perturb_params.append("".join(f"_{str(k)}-{str(v)}" for k, v in perturber_combo.items()))

        logger.info(f"Starting perturbation for {output_perturb_params[i]}")

        if isinstance(perturber, PerturbImage):
            perturber = FramewisePerturber(frame_perturber=perturber)

        maite_perturber = MAITEMultiobjectTrackingAugmentation(
            augment=perturber,
            augment_id=output_perturb_params[i],
        )

        params_dir = Path(staging_dir) / output_perturb_params[i].lstrip("_")

        sequences: list[VideoSequence] = []
        for idx in range(len(dataset)):
            stream, target, meta = dataset[idx]

            merged_meta: dict[str, Any] = {**meta}
            if metadata:
                merged_meta.update(metadata.get(meta["id"], {}))

            md = DatumMetadata(**merged_meta)
            aug_streams, aug_targets, _aug_md = maite_perturber(
                batch=(
                    [stream],
                    [target],
                    [md],
                ),
            )
            sequences.append(
                _to_datamaite_sequence(
                    aug_stream=aug_streams[0],
                    aug_target=aug_targets[0],
                    source_seq=source_sequences[idx],
                    staging_dir=params_dir,
                    uri_by_id=uri_by_id,
                    index2label=index2label,
                ),
            )

        augmented_datasets.append(
            BoxTrackDataset(
                sequences=tuple(sequences),
                categories=dict(dataset.categories),
                dataset_id=output_perturb_params[i],
                dataset_metadata=dataset.dataset_metadata,  # carry taxonomy/provenance through
            ),
        )
    return zip(output_perturb_params, augmented_datasets, strict=False)
