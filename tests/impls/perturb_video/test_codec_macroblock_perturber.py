from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from fractions import Fraction
from typing import ClassVar

import numpy as np
import pytest
from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from syrupy.assertion import SnapshotAssertion
from typing_extensions import override

from nrtk.impls.perturb_video import CodecMacroblockPerturber
from nrtk.impls.perturb_video._codec_macroblock_perturber import _DecodedVideoFrame
from nrtk.interfaces import PerturbVideo, VideoFrame
from tests.impls import INPUT_DRONE_VIDEO_FILE_PATH as INPUT_VIDEO_FILE_PATH
from tests.impls.perturb_video.perturber_tests_mixin import PerturbVideoTestsMixin
from tests.impls.perturb_video.test_perturber_utils import perturber_assertions
from tests.utils.video_io import read_video


def _make_random_frames(
    *,
    shape: tuple[int, ...] = (32, 32, 3),
    n: int = 5,
    fps: float = 30.0,
) -> Iterator[VideoFrame]:
    rng = np.random.default_rng(123)
    image = rng.integers(low=0, high=255, size=shape, dtype=np.uint8)
    return iter(
        [
            VideoFrame(
                # Shift base image to create distinct next frame
                # This prevents each frame from encoding as keyframe
                image=np.roll(image, shift=frame_i, axis=1),
                timestamp=frame_i / fps,
                boxes=[(AxisAlignedBoundingBox(min_vertex=(1, 1), max_vertex=(5, 5)), {"score": 1.0})],
                additional_params={"source": "test", "frame_index": frame_i},
            )
            for frame_i in range(n)
        ],
    )


@pytest.mark.pyav
class TestCodecMacroblockPerturber(PerturbVideoTestsMixin):
    impl_class: ClassVar[type[PerturbVideo]] = CodecMacroblockPerturber

    @override
    def make_perturber(self) -> PerturbVideo:
        """Return a codec perturber for shared video interface tests."""
        return CodecMacroblockPerturber(frame_rate=30.0)

    @override
    def make_frames(self) -> list[VideoFrame]:
        """Return fresh frames for shared video interface tests."""
        return list(_make_random_frames(n=3))

    def test_empty_input(self) -> None:
        """Verify an empty video produces no output frames."""
        inst = CodecMacroblockPerturber()
        assert list(inst(frames=iter([]))) == []

    def test_round_trip_preserves_video_contract(self) -> None:
        """Verify codec round trips preserve the video-frame contract."""
        frames = list(_make_random_frames())
        inst = CodecMacroblockPerturber(bit_rate=16_000, frame_rate=30.0)

        out_frames = list(perturber_assertions(perturb=inst, frames=iter(frames)))

        assert len(out_frames) == 5
        assert any(
            not np.array_equal(expected.image, actual.image)
            for expected, actual in zip(frames, out_frames, strict=True)
        )
        for frame in out_frames:
            assert frame.image.shape == (32, 32, 3)
            assert frame.image.dtype == np.uint8
            assert list(frame.boxes) == [
                (AxisAlignedBoundingBox(min_vertex=(1, 1), max_vertex=(5, 5)), {"score": 1.0}),
            ]
            assert frame.additional_params["source"] == "test"

    def test_quantizer_changes_output_strength(self) -> None:
        """Verify stronger quantization produces stronger perturbation."""
        frames = list(_make_random_frames(n=5))
        low_quantizer_frames = list(CodecMacroblockPerturber(quantizer=2, frame_rate=30.0)(frames=iter(frames)))
        high_quantizer_frames = list(CodecMacroblockPerturber(quantizer=31, frame_rate=30.0)(frames=iter(frames)))

        low_quantizer_mae = np.mean(
            [
                np.mean(np.abs(expected.image.astype(np.float32) - actual.image.astype(np.float32)))
                for expected, actual in zip(frames, low_quantizer_frames, strict=True)
            ],
        )
        high_quantizer_mae = np.mean(
            [
                np.mean(np.abs(expected.image.astype(np.float32) - actual.image.astype(np.float32)))
                for expected, actual in zip(frames, high_quantizer_frames, strict=True)
            ],
        )

        assert high_quantizer_mae > low_quantizer_mae

    def test_regression(self, ssim_tiff_snapshot: SnapshotAssertion, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression testing results to detect codec compression changes."""
        # FFmpeg's automatic thread count depends on the host and can change encoded output
        # Use single thread to remove this possibility
        monkeypatch.setattr(
            target=CodecMacroblockPerturber,
            name="_ENCODER_THREAD_COUNT",
            value=1,
        )

        # only using first 5 frames of video
        frames = list(read_video(INPUT_VIDEO_FILE_PATH))[:5]
        inst = CodecMacroblockPerturber(
            codec="mpeg4",
            quantizer=20,
            frame_rate=30.0,
            gop_size=5,
            max_b_frames=0,
        )

        out_frames = list(inst(frames=iter(frames)))

        # Snapshot frames 0, 2, and 4 concatenated
        # This keeps a single snapshot image while probing the entire video segment
        ssim_tiff_snapshot.assert_match(np.concatenate([out_frames[i].image for i in (0, 2, 4)], axis=1))

    @pytest.mark.parametrize(
        ("shape"),
        [
            (17, 19),
            (17, 19, 1),
            (17, 19, 3),
        ],
    )
    def test_preserves_odd_frame_shapes(self, shape: tuple[int, ...]) -> None:
        """Verify padded codec input is cropped back to the original shape."""
        frames = _make_random_frames(shape=shape, n=3)
        inst = CodecMacroblockPerturber(bit_rate=16_000, frame_rate=30.0)

        for frame in perturber_assertions(perturb=inst, frames=frames):
            assert frame.image.shape == shape
            assert frame.image.dtype == np.uint8

    @pytest.mark.parametrize(
        ("frames", "expectation"),
        [
            (
                iter([VideoFrame(image=np.ones((16, 16, 3), dtype=np.int16), timestamp=0.0)]),
                pytest.raises(NotImplementedError, match="supports uint8 and floating-point"),
            ),
            (
                iter([VideoFrame(image=np.ones((16, 16, 4), dtype=np.uint8), timestamp=0.0)]),
                pytest.raises(ValueError, match="single-channel, or RGB"),
            ),
        ],
    )
    def test_invalid_frame_images(self, frames: Iterator[VideoFrame], expectation: AbstractContextManager) -> None:
        """Verify unsupported frame dtypes and channel counts fail clearly."""
        inst = CodecMacroblockPerturber()
        with expectation:
            list(inst(frames=frames))

    @pytest.mark.parametrize("invalid_value", [-0.1, 1.1, np.nan, np.inf, -np.inf])
    def test_invalid_float_frame_values(self, invalid_value: float) -> None:
        """Verify floating-point frames must contain finite normalized values."""
        image = np.full((16, 16, 3), fill_value=0.5, dtype=np.float32)
        image[0, 0, 0] = invalid_value
        frames = iter([VideoFrame(image=image, timestamp=0.0)])

        with pytest.raises(ValueError, match=r"finite values in \[0, 1\]"):
            list(CodecMacroblockPerturber()(frames=frames))

    @pytest.mark.parametrize(
        ("timestamps", "error_match"),
        [
            ([0.0, 0.0], "strictly increasing"),
            ([1.0, 0.0], "strictly increasing"),
            ([0.0, np.nan], "finite"),
            ([0.0, np.inf], "finite"),
        ],
    )
    def test_invalid_timestamps_for_inferred_frame_rate(
        self,
        timestamps: list[float],
        error_match: str,
    ) -> None:
        """Verify invalid timestamps cannot produce a zero or non-finite frame rate."""
        frames = list(_make_random_frames(n=len(timestamps)))
        for frame, timestamp in zip(frames, timestamps, strict=True):
            frame.timestamp = timestamp

        with pytest.raises(ValueError, match=error_match):
            CodecMacroblockPerturber()._get_frame_rate(frames=frames)

    def test_inferred_frame_rate_uses_median_timestamp_delta(self) -> None:
        """Verify irregular positive timestamp intervals produce a stable inferred rate."""
        frames = list(_make_random_frames(n=4))
        for frame, timestamp in zip(frames, [0.0, 0.04, 0.08, 0.28], strict=True):
            frame.timestamp = timestamp

        assert CodecMacroblockPerturber()._get_frame_rate(frames=frames) == Fraction(numerator=25, denominator=1)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_float_frames_are_converted_through_uint8_and_restored(self, dtype: np.dtype) -> None:
        """Verify normalized float inputs are restored to their original dtype."""
        frames = [
            VideoFrame(
                image=(frame.image.astype(dtype) / np.array(255, dtype=dtype)).astype(dtype),
                timestamp=frame.timestamp,
                boxes=frame.boxes,
                additional_params=frame.additional_params,
            )
            for frame in _make_random_frames(n=3)
        ]
        inst = CodecMacroblockPerturber(quantizer=20, frame_rate=30.0)

        out_frames = list(perturber_assertions(perturb=inst, frames=iter(frames)))

        assert len(out_frames) == len(frames)
        for frame in out_frames:
            assert frame.image.dtype == dtype
            assert frame.image.shape == (32, 32, 3)
            assert np.all(frame.image >= 0.0)
            assert np.all(frame.image <= 1.0)

    @pytest.mark.parametrize(
        ("kwargs", "expectation"),
        [
            ({"codec": ""}, pytest.raises(ValueError, match="codec")),
            ({"bit_rate": 0}, pytest.raises(ValueError, match="bit_rate")),
            (
                {"min_bit_rate": 128_000, "max_bit_rate": 64_000},
                pytest.raises(ValueError, match="min_bit_rate"),
            ),
            (
                {"bit_rate_buffer_size": 64_000, "encoder_options": {"bufsize": "32k"}},
                pytest.raises(ValueError, match="rate-control"),
            ),
            (
                {"min_bit_rate": 32_000, "encoder_options": {"maxrate": "64k"}},
                pytest.raises(ValueError, match="rate-control"),
            ),
            ({"quantizer": 0}, pytest.raises(ValueError, match="quantizer")),
            ({"crf": -1}, pytest.raises(ValueError, match="crf")),
            (
                {"quantizer": 20, "crf": 32},
                pytest.raises(ValueError, match="Only one of quantizer, crf, or qp"),
            ),
            (
                {"quantizer": 20, "encoder_options": {"qmin": "4"}},
                pytest.raises(ValueError, match="quality-control"),
            ),
            (
                {"quantizer": 20, "encoder_options": {"crf": "32"}},
                pytest.raises(ValueError, match="quality-control"),
            ),
            ({"frame_rate": 0.0}, pytest.raises(ValueError, match="frame_rate")),
            ({"packet_loss_model": "bad_network"}, pytest.raises(ValueError, match="packet_loss_model")),
            ({"packet_loss_rate": -0.1}, pytest.raises(ValueError, match="packet_loss_rate")),
            ({"packet_loss_burst_length": 0}, pytest.raises(ValueError, match="packet_loss_burst_length")),
            ({"packet_loss_mode": "rtp"}, pytest.raises(ValueError, match="packet_loss_mode")),
            (
                {"gilbert_elliott_bad_loss_rate": 2.0},
                pytest.raises(ValueError, match="gilbert_elliott_bad_loss_rate"),
            ),
            ({"frame_count_policy": "repeat_last"}, pytest.raises(ValueError, match="frame_count_policy")),
            ({}, does_not_raise()),
        ],
    )
    def test_constructor_validation(self, kwargs: dict, expectation: AbstractContextManager) -> None:
        """Verify representative constructor validation cases."""
        with expectation:
            CodecMacroblockPerturber(**kwargs)

    def test_frame_count_mismatch_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify strict frame-count policy raises on decode count mismatch."""
        inst = CodecMacroblockPerturber()

        def fake_encode(
            *,
            frames: list[VideoFrame],
            frame_rate: Fraction,
            container_format: str | None,
        ) -> list[VideoFrame]:
            del frame_rate, container_format
            return frames

        def fake_decode(**_: object) -> list[VideoFrame]:
            return []

        monkeypatch.setattr(target=inst, name="_encode", value=fake_encode)
        monkeypatch.setattr(target=inst, name="_decode_frames", value=fake_decode)

        with pytest.raises(RuntimeError, match="Decoded frame count"):
            list(inst(frames=_make_random_frames(n=2)))

    def test_configuration(self) -> None:
        """Verify configuration round trips through SMQTK helpers."""
        inst = CodecMacroblockPerturber(
            codec="mpeg4",
            container_format="mp4",
            bit_rate=32_000,
            min_bit_rate=16_000,
            max_bit_rate=32_000,
            bit_rate_buffer_size=64_000,
            quantizer=20,
            crf=None,
            qp=None,
            pixel_format="yuv420p",
            frame_rate=15.0,
            gop_size=8,
            max_b_frames=0,
            packet_loss_model="gilbert_elliott",
            packet_loss_rate=0.25,
            packet_loss_burst_length=3,
            packet_loss_mode="transport_stream",
            packet_loss_seed=123,
            packet_loss_preserve_keyframes=False,
            gilbert_elliott_good_loss_rate=0.01,
            gilbert_elliott_bad_loss_rate=0.9,
            gilbert_elliott_good_to_bad_rate=0.02,
            gilbert_elliott_bad_to_good_rate=0.5,
            transport_loss_preserve_payload_starts=False,
            frame_count_policy="black",
            encoder_options={"strict": "experimental"},
        )
        for hydrated in configuration_test_helper(inst):
            assert hydrated.codec == "mpeg4"
            assert hydrated.container_format == "mp4"
            assert hydrated.bit_rate == 32_000
            assert hydrated.min_bit_rate == 16_000
            assert hydrated.max_bit_rate == 32_000
            assert hydrated.bit_rate_buffer_size == 64_000
            assert hydrated.quantizer == 20
            assert hydrated.crf is None
            assert hydrated.qp is None
            assert hydrated.pixel_format == "yuv420p"
            assert hydrated.frame_rate == 15.0
            assert hydrated.gop_size == 8
            assert hydrated.max_b_frames == 0
            assert hydrated.packet_loss_model == "gilbert_elliott"
            assert hydrated.packet_loss_rate == 0.25
            assert hydrated.packet_loss_burst_length == 3
            assert hydrated.packet_loss_mode == "transport_stream"
            assert hydrated.packet_loss_seed == 123
            assert hydrated.packet_loss_preserve_keyframes is False
            assert hydrated.gilbert_elliott_good_loss_rate == 0.01
            assert hydrated.gilbert_elliott_bad_loss_rate == 0.9
            assert hydrated.gilbert_elliott_good_to_bad_rate == 0.02
            assert hydrated.gilbert_elliott_bad_to_good_rate == 0.5
            assert hydrated.transport_loss_preserve_payload_starts is False
            assert hydrated.frame_count_policy == "black"
            assert hydrated.encoder_options == {"strict": "experimental"}

    def test_get_config_copies_encoder_options(self) -> None:
        """Verify returned encoder options do not alias instance state."""
        inst = CodecMacroblockPerturber(encoder_options={"strict": "experimental"})
        cfg = inst.get_config()

        cfg["encoder_options"]["strict"] = "changed"

        assert inst.encoder_options == {"strict": "experimental"}

    def test_packet_loss_seed_reproducibility(self) -> None:
        """Verify seeded packet loss is reproducible without relying on codec byte snapshots."""
        frames = list(_make_random_frames(n=12))
        baseline = list(
            CodecMacroblockPerturber(
                frame_rate=30.0,
                gop_size=12,
                frame_count_policy="black",
            )(frames=iter(frames)),
        )
        first_run = list(
            CodecMacroblockPerturber(
                frame_rate=30.0,
                gop_size=12,
                packet_loss_rate=0.5,
                packet_loss_seed=123,
                frame_count_policy="black",
            )(frames=iter(frames)),
        )
        second_run = list(
            CodecMacroblockPerturber(
                frame_rate=30.0,
                gop_size=12,
                packet_loss_rate=0.5,
                packet_loss_seed=123,
                frame_count_policy="black",
            )(frames=iter(frames)),
        )

        assert [frame.timestamp for frame in first_run] == [frame.timestamp for frame in second_run]
        for first_frame, second_frame in zip(first_run, second_run, strict=True):
            assert np.array_equal(first_frame.image, second_frame.image)
        assert any(
            not np.array_equal(expected.image, actual.image)
            for expected, actual in zip(baseline, first_run, strict=True)
        )

    def test_black_fill_preserves_decoded_frame_positions(self) -> None:
        """Verify black infill keeps surviving frames aligned by decoded frame index."""
        inst = CodecMacroblockPerturber(frame_count_policy="black")
        decoded_images = inst._decoded_images_for_policy(
            decoded_frames=[
                _DecodedVideoFrame(image=np.full((4, 4, 3), fill_value=100, dtype=np.uint8), frame_index=1),
                _DecodedVideoFrame(image=np.full((4, 4, 3), fill_value=200, dtype=np.uint8), frame_index=3),
            ],
            input_frames=list(_make_random_frames(shape=(4, 4, 3), n=5)),
        )

        assert len(decoded_images) == 5
        assert np.all(decoded_images[0] == 0)
        assert np.all(decoded_images[1] == 100)
        assert np.all(decoded_images[2] == 0)
        assert np.all(decoded_images[3] == 200)
        assert np.all(decoded_images[4] == 0)

    def test_packet_loss(self) -> None:
        """Verify packet loss preserves the video-frame contract."""
        frames = list(_make_random_frames(n=12))
        baseline = list(
            CodecMacroblockPerturber(
                frame_rate=30.0,
                gop_size=12,
                frame_count_policy="black",
            )(frames=iter(frames)),
        )
        inst = CodecMacroblockPerturber(
            frame_rate=30.0,
            gop_size=12,
            packet_loss_rate=1.0,
            packet_loss_seed=123,
            frame_count_policy="black",
        )

        out_frames = list(inst(frames=iter(frames)))

        assert len(out_frames) == len(frames)
        assert any(
            not np.array_equal(expected.image, actual.image)
            for expected, actual in zip(baseline, out_frames, strict=True)
        )
        for expected, actual in zip(frames, out_frames, strict=True):
            assert actual.image.shape == expected.image.shape
            assert actual.image.dtype == np.uint8
            assert actual.timestamp == expected.timestamp
            assert actual.additional_params == expected.additional_params

    def test_total_compressed_packet_loss_raises(self) -> None:
        """Verify an entirely undecodable compressed stream raises a contextual error."""
        frames = list(_make_random_frames(n=12))
        inst = CodecMacroblockPerturber(
            frame_rate=30.0,
            gop_size=12,
            packet_loss_rate=1.0,
            packet_loss_preserve_keyframes=False,
            frame_count_policy="black",
        )

        with pytest.raises(RuntimeError, match="encoded or damaged video stream") as ex_info:
            list(inst(frames=iter(frames)))

        assert ex_info.value.__cause__ is not None
        assert type(ex_info.value.__cause__).__name__ == "InvalidDataError"

    def test_transport_stream_packet_loss(self) -> None:
        """Verify transport-stream loss preserves the video contract."""
        frames = list(_make_random_frames(n=12))
        inst = CodecMacroblockPerturber(
            frame_rate=30.0,
            gop_size=2,
            packet_loss_rate=0.5,
            packet_loss_mode="transport_stream",
            packet_loss_seed=123,
            frame_count_policy="black",
        )

        out_frames = list(inst(frames=iter(frames)))

        assert len(out_frames) == len(frames)
        for expected, actual in zip(frames, out_frames, strict=True):
            assert actual.image.shape == expected.image.shape
            assert actual.image.dtype == np.uint8
            assert actual.timestamp == expected.timestamp
