"""Codec-backed video perturber for simulating macroblocking artifacts."""

from __future__ import annotations

__all__ = ["CodecMacroblockPerturber"]

import io
from collections.abc import Generator, Iterable, Iterator
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, BinaryIO, Literal

import av
import numpy as np
from av.error import InvalidDataError
from av.video.frame import VideoFrame as AVVideoFrame
from av.video.reformatter import VideoReformatter
from typing_extensions import override

from nrtk.impls.perturb_video._mpegts import (
    _MPEGTS_PACKET_SIZE,
    _mpegts_payload_unit_start,
    _mpegts_pid,
    _mpegts_table_pids,
)
from nrtk.interfaces import PerturbVideo, VideoFrame
from nrtk.interfaces._perturb_video import _perturb_guard


@dataclass(frozen=True)
class _DecodedVideoFrame:
    image: np.ndarray[Any, Any]
    frame_index: int


@dataclass
class _PacketLossState:
    remaining_packets: int = 0
    gilbert_elliott_bad_state: bool = False


class CodecMacroblockPerturber(PerturbVideo):
    """Round-trip video frames through a real codec using stressed compression settings.

    This perturber encodes the input frame sequence to an in-memory compressed video stream,
    decodes it back to arrays, and yields frames aligned by position with the input frames.
    It can optionally drop encoded packets before muxing to approximate compressed-packet loss.
    It currently supports ``uint8`` and normalized floating-point grayscale and RGB frame arrays.
    """

    # FFmpeg thread count will generally be set to 0 (auto mode) in production
    # Defining and using this variable allows this to be overwritten for testing
    _ENCODER_THREAD_COUNT = 0

    def __init__(
        self,
        *,
        codec: str = "mpeg4",
        container_format: str = "mp4",
        pixel_format: str = "yuv420p",
        frame_rate: float | None = None,
        quantizer: int | None = None,
        crf: int | None = None,
        qp: int | None = None,
        bit_rate: int | None = None,
        min_bit_rate: int | None = None,
        max_bit_rate: int | None = None,
        bit_rate_buffer_size: int | None = None,
        gop_size: int = 12,
        max_b_frames: int = 0,
        encoder_options: dict[str, str] | None = None,
        packet_loss_mode: Literal["compressed_packet", "transport_stream"] = "compressed_packet",
        packet_loss_model: Literal["independent", "gilbert_elliott"] = "independent",
        packet_loss_rate: float = 0.0,
        packet_loss_burst_length: int = 1,
        packet_loss_seed: int | None = None,
        packet_loss_preserve_keyframes: bool = True,
        transport_loss_preserve_payload_starts: bool = True,
        gilbert_elliott_good_loss_rate: float = 0.001,
        gilbert_elliott_bad_loss_rate: float = 0.8,
        gilbert_elliott_good_to_bad_rate: float = 0.01,
        gilbert_elliott_bad_to_good_rate: float = 0.25,
        frame_count_policy: Literal["strict", "black"] = "strict",
    ) -> None:
        """Initialize the codec-backed macroblocking perturber.

        Args:
            codec:
                Encoder name to use through PyAV/FFmpeg.
            container_format:
                Container format for the in-memory round trip.
            pixel_format:
                Codec pixel format to use for encoding.
            frame_rate:
                Frame rate for the encoded stream. If ``None``, infer from input timestamps.
            quantizer:
                Fixed FFmpeg ``qmin`` / ``qmax`` value on range [1, 31], where
                larger values apply stronger compression. This is reliable for
                the default ``mpeg4`` encoder, but support is codec-dependent.
            crf:
                Optional constant rate factor value, passed as FFmpeg ``crf``.
                Larger values generally apply stronger compression, but support
                and valid ranges are codec-dependent.
            qp:
                Optional fixed quantization parameter value, passed as FFmpeg
                ``qp``. Larger values generally apply stronger compression, but
                support and valid ranges are codec-dependent.
            bit_rate:
                Optional target encoder bit rate in bits per second. If ``None``,
                do not set the encoder target bit rate.
            min_bit_rate:
                Optional encoder minimum bit rate in bits per second, passed as
                FFmpeg ``minrate``.
            max_bit_rate:
                Optional encoder maximum bit rate in bits per second, passed as
                FFmpeg ``maxrate``.
            bit_rate_buffer_size:
                Optional rate-control buffer size in bits, passed as FFmpeg
                ``bufsize``.
            gop_size:
                Encoder group-of-pictures size.
            max_b_frames:
                Maximum number of B-frames. Defaults to 0 for simpler frame count preservation.
            encoder_options:
                Additional codec-specific encoder options.
            packet_loss_mode:
                Packet-loss simulation mode. ``"compressed_packet"`` drops whole
                encoded codec packets before muxing, which can produce missing
                frames or dependency-chain artifacts. ``"transport_stream"``
                muxes to MPEG-TS and drops 188-byte transport packets, which can
                damage parts of encoded frames and trigger decoder concealment.
                This is closer to packetized loss, but not a full RTP/UDP
                network simulator.
            packet_loss_model:
                Packet-loss process. ``"independent"`` uses ``packet_loss_rate`` and
                ``packet_loss_burst_length``. ``"gilbert_elliott"`` uses the
                Gilbert-Elliott good/bad-state probabilities.
            packet_loss_rate:
                Probability of starting a packet-loss event. This simulates compressed packet loss by
                default. When ``packet_loss_mode`` is ``"transport_stream"``, this is the probability
                of starting a loss event on eligible MPEG-TS media packets after muxing. Only used
                when ``packet_loss_model`` is ``"independent"``.
            packet_loss_burst_length:
                Number of consecutive eligible packets to drop per loss event. The default value of
                1 preserves independent packet-loss behavior. Only used when ``packet_loss_model``
                is ``"independent"``.
            packet_loss_seed:
                Optional seed for deterministic packet-loss sampling.
            packet_loss_preserve_keyframes:
                If ``True``, encoded key packets are not dropped.
                This can reduce complete decoder failure and keep loss effects
                focused on dependent frames.
            transport_loss_preserve_payload_starts:
                If ``True``, MPEG-TS packets that start payload units are not dropped.
                This can reduce complete decoded frame drops while still damaging
                continuation payload data.
            gilbert_elliott_good_loss_rate:
                Packet-loss probability while the Gilbert-Elliott channel is in the good state.
            gilbert_elliott_bad_loss_rate:
                Packet-loss probability while the Gilbert-Elliott channel is in the bad state.
            gilbert_elliott_good_to_bad_rate:
                Probability of transitioning from the good state to the bad state after a packet.
            gilbert_elliott_bad_to_good_rate:
                Probability of transitioning from the bad state to the good state after a packet.
            frame_count_policy:
                Policy for decoded frame-count mismatches. ``"strict"`` raises,
                while ``"black"`` fills missing decoded frame positions with black frames.
        """
        super().__init__()

        self._validate_codec_settings(
            codec=codec,
            container_format=container_format,
            pixel_format=pixel_format,
            frame_rate=frame_rate,
            gop_size=gop_size,
            max_b_frames=max_b_frames,
        )
        self._validate_quality_settings(quantizer=quantizer, crf=crf, qp=qp, encoder_options=encoder_options)
        self._validate_rate_control(
            bit_rate=bit_rate,
            min_bit_rate=min_bit_rate,
            max_bit_rate=max_bit_rate,
            bit_rate_buffer_size=bit_rate_buffer_size,
            encoder_options=encoder_options,
        )
        self._validate_packet_loss_settings(
            packet_loss_mode=packet_loss_mode,
            packet_loss_model=packet_loss_model,
            packet_loss_rate=packet_loss_rate,
            packet_loss_burst_length=packet_loss_burst_length,
            gilbert_elliott_good_loss_rate=gilbert_elliott_good_loss_rate,
            gilbert_elliott_bad_loss_rate=gilbert_elliott_bad_loss_rate,
            gilbert_elliott_good_to_bad_rate=gilbert_elliott_good_to_bad_rate,
            gilbert_elliott_bad_to_good_rate=gilbert_elliott_bad_to_good_rate,
            frame_count_policy=frame_count_policy,
        )

        self.codec = codec
        self.container_format = container_format
        self.pixel_format = pixel_format
        self.frame_rate = frame_rate
        self.quantizer = quantizer
        self.crf = crf
        self.qp = qp
        self.bit_rate = bit_rate
        self.min_bit_rate = min_bit_rate
        self.max_bit_rate = max_bit_rate
        self.bit_rate_buffer_size = bit_rate_buffer_size
        self.gop_size = gop_size
        self.max_b_frames = max_b_frames
        self.encoder_options: dict[str, str] = dict(encoder_options) if encoder_options is not None else {}
        self.packet_loss_mode = packet_loss_mode
        self.packet_loss_model = packet_loss_model
        self.packet_loss_rate = packet_loss_rate
        self.packet_loss_burst_length = packet_loss_burst_length
        self.packet_loss_seed = packet_loss_seed
        self.packet_loss_preserve_keyframes = packet_loss_preserve_keyframes
        self.transport_loss_preserve_payload_starts = transport_loss_preserve_payload_starts
        self.gilbert_elliott_good_loss_rate = gilbert_elliott_good_loss_rate
        self.gilbert_elliott_bad_loss_rate = gilbert_elliott_bad_loss_rate
        self.gilbert_elliott_good_to_bad_rate = gilbert_elliott_good_to_bad_rate
        self.gilbert_elliott_bad_to_good_rate = gilbert_elliott_bad_to_good_rate
        self.frame_count_policy = frame_count_policy

    @override
    @_perturb_guard
    def perturb(
        self,
        *,
        frames: Iterator[VideoFrame],
        **_: Any,
    ) -> Generator[VideoFrame, None, None]:
        """Perturb video frames by encoding and decoding them with a lossy codec.

        Args:
            frames:
                Iterator over input video frames.

        Yields:
            Perturbed video frames in input-frame order.

        Raises:
            RuntimeError:
                If the encoded stream cannot be decoded or the strict frame-count policy detects a mismatch.
        """
        input_frames = list(frames)
        if len(input_frames) == 0:
            return

        self._validate_frame_images(frames=input_frames)

        frame_rate = self._get_frame_rate(frames=input_frames)
        container_format = "mpegts" if self.packet_loss_mode == "transport_stream" else self.container_format
        encoded = self._encode(frames=input_frames, frame_rate=frame_rate, container_format=container_format)
        if self.packet_loss_mode == "transport_stream":
            encoded = self._apply_transport_stream_packet_loss(file=encoded)
        decoded_frames = self._decode_frames(file=encoded, frame_rate=frame_rate, container_format=container_format)

        decoded_images = self._decoded_images_for_policy(decoded_frames=decoded_frames, input_frames=input_frames)

        if len(decoded_images) != len(input_frames):
            raise RuntimeError(
                f"Decoded frame count ({len(decoded_images)}) does not match input frame count ({len(input_frames)}).",
            )

        for frame, decoded_image in zip(input_frames, decoded_images, strict=True):
            yield VideoFrame(
                image=self._restore_image(image=decoded_image, original_image=frame.image),
                timestamp=frame.timestamp,
                boxes=deepcopy(frame.boxes),
                additional_params=deepcopy(frame.additional_params),
            )

    @staticmethod
    def _validate_codec_settings(  # noqa: C901 - simple constructor validation is clearer inline
        *,
        codec: str,
        container_format: str,
        pixel_format: str,
        frame_rate: float | None,
        gop_size: int,
        max_b_frames: int,
    ) -> None:
        if not codec:
            raise ValueError("codec must be a non-empty string.")
        if not container_format:
            raise ValueError("container_format must be a non-empty string.")
        if not pixel_format:
            raise ValueError("pixel_format must be a non-empty string.")
        if frame_rate is not None and frame_rate <= 0.0:
            raise ValueError("frame_rate must be > 0 when provided.")
        if gop_size <= 0:
            raise ValueError("gop_size must be > 0.")
        if max_b_frames < 0:
            raise ValueError("max_b_frames must be >= 0.")

    @staticmethod
    def _validate_quality_settings(  # noqa: C901 - quality validation is clearer without one-line helpers
        *,
        quantizer: int | None,
        crf: int | None,
        qp: int | None,
        encoder_options: dict[str, str] | None,
    ) -> None:
        if sum(value is not None for value in (quantizer, crf, qp)) > 1:
            raise ValueError("Only one of quantizer, crf, or qp may be provided.")
        if quantizer is not None and not 1 <= quantizer <= 31:
            raise ValueError("quantizer must be in [1, 31] when provided.")
        if crf is not None and crf < 0:
            raise ValueError("crf must be >= 0 when provided.")
        if qp is not None and qp < 0:
            raise ValueError("qp must be >= 0 when provided.")
        if (
            encoder_options is not None
            and any(value is not None for value in (quantizer, crf, qp))
            and any(option in encoder_options for option in {"qmin", "qmax", "crf", "qp"})
        ):
            raise ValueError(
                "Explicit quality-control parameters cannot be combined with encoder_options containing "
                "qmin, qmax, crf, or qp.",
            )

    @staticmethod
    def _validate_rate_control(  # noqa: C901 - rate-control validation is a flat parameter check
        *,
        bit_rate: int | None,
        min_bit_rate: int | None,
        max_bit_rate: int | None,
        bit_rate_buffer_size: int | None,
        encoder_options: dict[str, str] | None,
    ) -> None:
        if bit_rate is not None and bit_rate <= 0:
            raise ValueError("bit_rate must be > 0 when provided.")
        if min_bit_rate is not None and min_bit_rate <= 0:
            raise ValueError("min_bit_rate must be > 0 when provided.")
        if max_bit_rate is not None and max_bit_rate <= 0:
            raise ValueError("max_bit_rate must be > 0 when provided.")
        if bit_rate_buffer_size is not None and bit_rate_buffer_size <= 0:
            raise ValueError("bit_rate_buffer_size must be > 0 when provided.")
        if min_bit_rate is not None and max_bit_rate is not None and min_bit_rate > max_bit_rate:
            raise ValueError("min_bit_rate must be <= max_bit_rate when both are provided.")
        if (
            encoder_options is not None
            and any(value is not None for value in (min_bit_rate, max_bit_rate, bit_rate_buffer_size))
            and any(option in encoder_options for option in {"minrate", "maxrate", "bufsize"})
        ):
            raise ValueError(
                "Explicit rate-control parameters cannot be combined with encoder_options containing "
                "minrate, maxrate, or bufsize.",
            )

    @staticmethod
    def _validate_packet_loss_settings(  # noqa: C901 - packet-loss validation is a flat parameter check
        *,
        packet_loss_mode: str,
        packet_loss_model: str,
        packet_loss_rate: float,
        packet_loss_burst_length: int,
        gilbert_elliott_good_loss_rate: float,
        gilbert_elliott_bad_loss_rate: float,
        gilbert_elliott_good_to_bad_rate: float,
        gilbert_elliott_bad_to_good_rate: float,
        frame_count_policy: str,
    ) -> None:
        if packet_loss_mode not in ("compressed_packet", "transport_stream"):
            raise ValueError("packet_loss_mode must be one of: 'compressed_packet' or 'transport_stream'.")
        if packet_loss_model not in ("independent", "gilbert_elliott"):
            raise ValueError("packet_loss_model must be one of: 'independent' or 'gilbert_elliott'.")
        CodecMacroblockPerturber._validate_probability(name="packet_loss_rate", value=packet_loss_rate)
        if packet_loss_burst_length <= 0:
            raise ValueError("packet_loss_burst_length must be > 0.")
        for name, value in {
            "gilbert_elliott_good_loss_rate": gilbert_elliott_good_loss_rate,
            "gilbert_elliott_bad_loss_rate": gilbert_elliott_bad_loss_rate,
            "gilbert_elliott_good_to_bad_rate": gilbert_elliott_good_to_bad_rate,
            "gilbert_elliott_bad_to_good_rate": gilbert_elliott_bad_to_good_rate,
        }.items():
            CodecMacroblockPerturber._validate_probability(name=name, value=value)
        if frame_count_policy not in ("strict", "black"):
            raise ValueError("frame_count_policy must be one of: 'strict' or 'black'.")

    @staticmethod
    def _validate_frame_images(*, frames: list[VideoFrame]) -> None:
        for frame in frames:
            # The PyAV path below encodes as rgb24, so inputs must be representable as uint8 RGB
            # Normalized floats have a clear conversion policy, other dtypes do not
            is_floating = np.issubdtype(frame.image.dtype, np.floating)
            if frame.image.dtype != np.uint8 and not is_floating:
                raise NotImplementedError(
                    "CodecMacroblockPerturber currently supports uint8 and floating-point frame images.",
                )
            if is_floating and (
                not np.all(np.isfinite(frame.image)) or np.any(frame.image < 0.0) or np.any(frame.image > 1.0)
            ):
                raise ValueError(
                    "Floating-point frame images must contain only finite values in [0, 1].",
                )
            if frame.image.ndim == 2:
                continue
            CodecMacroblockPerturber._validate_three_dimensional_frame_shape(image=frame.image)

    @staticmethod
    def _validate_three_dimensional_frame_shape(*, image: np.ndarray[Any, Any]) -> None:
        if image.ndim != 3:
            raise ValueError(f"Frame image must have 2 or 3 dimensions; instead given {image.ndim}.")
        if image.shape[2] not in (1, 3):
            raise ValueError(
                f"CodecMacroblockPerturber supports 2D grayscale, single-channel, or RGB frames; got "
                f"{image.shape[2]} channels.",
            )

    @staticmethod
    def _fill_missing_frames_with_black(
        *,
        decoded_frames: list[_DecodedVideoFrame],
        input_frame_count: int,
        frame_shape: tuple[int, ...],
    ) -> list[np.ndarray[Any, Any]]:
        filled_images: list[np.ndarray[Any, Any] | None] = [None] * input_frame_count
        for decoded_frame in decoded_frames:
            if 0 <= decoded_frame.frame_index < input_frame_count and filled_images[decoded_frame.frame_index] is None:
                filled_images[decoded_frame.frame_index] = decoded_frame.image

        # Fill missing indices with black frames
        return [image if image is not None else np.zeros(frame_shape, dtype=np.uint8) for image in filled_images]

    def _decoded_images_for_policy(
        self,
        *,
        decoded_frames: list[_DecodedVideoFrame],
        input_frames: list[VideoFrame],
    ) -> list[np.ndarray[Any, Any]]:
        if self.frame_count_policy == "strict":
            return [frame.image for frame in decoded_frames]
        frame_shape = self._prepare_image_for_encoding(input_frames[0].image).shape
        return CodecMacroblockPerturber._fill_missing_frames_with_black(
            decoded_frames=decoded_frames,
            input_frame_count=len(input_frames),
            frame_shape=frame_shape,
        )

    def _encode(
        self,
        *,
        frames: list[VideoFrame],
        frame_rate: Fraction,
        container_format: str | None = None,
    ) -> io.BytesIO:
        buffer = io.BytesIO()
        first_image = self._prepare_image_for_encoding(frames[0].image)
        packet_loss_rng = np.random.default_rng(self.packet_loss_seed)
        packet_loss_state = _PacketLossState()
        container_format = container_format or self.container_format

        video = av.open(buffer, mode="w", format=container_format)
        try:
            video_stream: Any = video.add_stream(self.codec, rate=frame_rate, options=self._encoder_options())
            time_base = Fraction(numerator=frame_rate.denominator, denominator=frame_rate.numerator)
            self._configure_video_stream(
                video_stream=video_stream,
                width=first_image.shape[1],
                height=first_image.shape[0],
                time_base=time_base,
            )

            frame_converter = VideoReformatter()
            self._encode_and_mux_frames(
                frames=frames,
                video=video,
                video_stream=video_stream,
                frame_converter=frame_converter,
                time_base=time_base,
                packet_loss_rng=packet_loss_rng,
                packet_loss_state=packet_loss_state,
            )
            # Flush delayed packets after all frames have been submitted
            self._mux_packets(
                packets=video_stream.encode(),
                video=video,
                packet_loss_rng=packet_loss_rng,
                packet_loss_state=packet_loss_state,
            )
        finally:
            video.close()

        buffer.seek(0)
        return buffer

    def _encode_and_mux_frames(
        self,
        *,
        frames: list[VideoFrame],
        video: Any,  # noqa: ANN401
        video_stream: Any,  # noqa: ANN401
        frame_converter: VideoReformatter,
        time_base: Fraction,
        packet_loss_rng: np.random.Generator,
        packet_loss_state: _PacketLossState,
    ) -> None:
        for frame_i, frame in enumerate(frames):
            frame_rgb = AVVideoFrame.from_ndarray(
                array=self._prepare_image_for_encoding(frame.image),
                format="rgb24",
            )
            frame_rgb.pts = frame_i
            frame_rgb.time_base = time_base
            # Convert rgb24 input into the encoder pixel format, yuv420p by default
            frame_yuv = frame_converter.reformat(frame_rgb, format=video_stream.format.name)
            self._mux_packets(
                packets=video_stream.encode(frame_yuv),
                video=video,
                packet_loss_rng=packet_loss_rng,
                packet_loss_state=packet_loss_state,
            )

    def _mux_packets(
        self,
        *,
        packets: Iterable[Any],
        video: Any,  # noqa: ANN401
        packet_loss_rng: np.random.Generator,
        packet_loss_state: _PacketLossState,
    ) -> None:
        for packet in packets:
            if not self._should_drop_compressed_packet(
                packet=packet,
                rng=packet_loss_rng,
                state=packet_loss_state,
            ):
                video.mux(packet)

    def _configure_video_stream(
        self,
        *,
        video_stream: Any,  # noqa: ANN401
        width: int,
        height: int,
        time_base: Fraction,
    ) -> None:
        video_stream.width = width
        video_stream.height = height
        video_stream.pix_fmt = self.pixel_format
        if self.bit_rate is not None:
            # Some PyAV stream attributes are read-only for some codecs/builds
            # The codec context assignment below is the primary bitrate setting
            with suppress(AttributeError, TypeError):  # noqa: FKA100 - suppress requires positional exception types
                video_stream.bit_rate = self.bit_rate
            video_stream.codec_context.bit_rate = self.bit_rate
        # Some PyAV builds do not allow setting time_base on the stream
        # The codec_context.time_base setting below should suffice in these cases
        with suppress(AttributeError, TypeError):  # noqa: FKA100 - suppress requires positional exception types
            video_stream.time_base = time_base
        video_stream.codec_context.time_base = time_base
        video_stream.codec_context.pix_fmt = self.pixel_format
        video_stream.codec_context.gop_size = self.gop_size
        video_stream.codec_context.max_b_frames = self.max_b_frames
        video_stream.codec_context.thread_count = self._ENCODER_THREAD_COUNT
        video_stream.codec_context.thread_type = "AUTO"

    def _should_drop_compressed_packet(
        self,
        *,
        packet: Any,  # noqa: ANN401
        rng: np.random.Generator,
        state: _PacketLossState | None = None,
    ) -> bool:
        """Apply compressed-packet-specific loss policy before generic loss sampling."""
        if self.packet_loss_mode != "compressed_packet" or not self._packet_loss_enabled():
            return False

        packet_loss_state = state or _PacketLossState()
        if self.packet_loss_preserve_keyframes and getattr(packet, "is_keyframe", False):
            return self._should_drop_loss_packet(rng=rng, state=packet_loss_state, eligible=False)

        return self._should_drop_loss_packet(rng=rng, state=packet_loss_state, eligible=True)

    def _apply_transport_stream_packet_loss(self, *, file: BinaryIO) -> io.BytesIO:
        # MPEG-TS is a sequence of fixed 188-byte packets. Byte 0 is the sync byte
        # The packet PID is a 13-bit field: the low 5 bits of byte 1 followed by byte 2
        # Byte 1 also contains the payload-unit-start bit, which marks table/PES payload starts
        # PAT/SDT/PMT are metadata tables needed for demuxing, PES packets carry encoded media payload
        file.seek(0)
        if not self._packet_loss_enabled():
            return io.BytesIO(file.read())

        stream = file.read()
        table_pids = _mpegts_table_pids(stream=stream)
        damaged = self._drop_mpegts_media_packets(stream=stream, table_pids=table_pids)

        result = io.BytesIO(bytes(damaged))
        result.seek(0)
        return result

    def _drop_mpegts_media_packets(self, *, stream: bytes, table_pids: set[int]) -> bytearray:
        rng = np.random.default_rng(self.packet_loss_seed)
        packet_loss_state = _PacketLossState()

        # Second pass: apply packet loss while preserving the table PIDs discovered above
        damaged = bytearray()
        for packet_start in range(0, len(stream), _MPEGTS_PACKET_SIZE):
            packet = stream[packet_start : packet_start + _MPEGTS_PACKET_SIZE]
            if len(packet) != _MPEGTS_PACKET_SIZE:
                # Keep any trailing partial data unchanged instead of trying to parse it as TS
                damaged.extend(packet)
                continue

            if not self._should_drop_loss_packet(
                rng=rng,
                state=packet_loss_state,
                eligible=not self._preserve_mpegts_packet(packet=packet, table_pids=table_pids),
            ):
                damaged.extend(packet)
        return damaged

    def _preserve_mpegts_packet(self, *, packet: bytes, table_pids: set[int]) -> bool:
        if _mpegts_pid(packet=packet) in table_pids:
            return True
        return self.transport_loss_preserve_payload_starts and _mpegts_payload_unit_start(packet)

    def _should_drop_loss_packet(
        self,
        *,
        rng: Any,  # noqa: ANN401 - accepts numpy generators and small test doubles
        state: _PacketLossState,
        eligible: bool,
    ) -> bool:
        """Sample whether an eligible packet-like unit is lost under the configured loss model."""
        if not self._packet_loss_enabled():
            return False

        if self.packet_loss_model == "gilbert_elliott":
            return self._should_drop_gilbert_elliott_packet(rng=rng, state=state, eligible=eligible)

        return self._should_drop_independent_packet(rng=rng, state=state, eligible=eligible)

    def _should_drop_independent_packet(
        self,
        *,
        rng: Any,  # noqa: ANN401
        state: _PacketLossState,
        eligible: bool,
    ) -> bool:
        # Independent loss counts only eligible packets, Gilbert-Elliott state advances above even when preserved
        if not eligible:
            return False

        if state.remaining_packets > 0:
            state.remaining_packets -= 1
            return True

        if not self._sample_probability(rng=rng, probability=self.packet_loss_rate):
            return False

        state.remaining_packets = self.packet_loss_burst_length - 1
        return True

    def _should_drop_gilbert_elliott_packet(
        self,
        *,
        rng: Any,  # noqa: ANN401
        state: _PacketLossState,
        eligible: bool,
    ) -> bool:
        if state.gilbert_elliott_bad_state:
            loss_rate = self.gilbert_elliott_bad_loss_rate
            transition_rate = self.gilbert_elliott_bad_to_good_rate
        else:
            loss_rate = self.gilbert_elliott_good_loss_rate
            transition_rate = self.gilbert_elliott_good_to_bad_rate

        packet_lost = self._sample_probability(rng=rng, probability=loss_rate)
        if self._sample_probability(rng=rng, probability=transition_rate):
            state.gilbert_elliott_bad_state = not state.gilbert_elliott_bad_state
        return eligible and packet_lost

    def _packet_loss_enabled(self) -> bool:
        if self.packet_loss_model == "independent":
            return self.packet_loss_rate > 0.0

        return self.gilbert_elliott_good_loss_rate > 0.0 or self.gilbert_elliott_bad_loss_rate > 0.0

    @staticmethod
    def _sample_probability(*, rng: Any, probability: float) -> bool:  # noqa: ANN401
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        return bool(rng.random() < probability)

    @staticmethod
    def _validate_probability(*, name: str, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")

    def _encoder_options(self) -> dict[str, str]:  # noqa: C901 - flat option assembly is clearer inline
        options = dict(self.encoder_options)
        if self.min_bit_rate is not None:
            options["minrate"] = str(self.min_bit_rate)
        if self.max_bit_rate is not None:
            options["maxrate"] = str(self.max_bit_rate)
        if self.bit_rate_buffer_size is not None:
            options["bufsize"] = str(self.bit_rate_buffer_size)
        if self.quantizer is not None:
            quantizer = str(self.quantizer)
            # qscale/global_quality do not reliably vary PyAV mpeg4 output, fixed qmin/qmax does
            options["qmin"] = quantizer
            options["qmax"] = quantizer
        if self.crf is not None:
            options["crf"] = str(self.crf)
        if self.qp is not None:
            options["qp"] = str(self.qp)
        return options

    def _decode_frames(
        self,
        *,
        file: BinaryIO,
        frame_rate: Fraction | None,
        container_format: str | None,
    ) -> list[_DecodedVideoFrame]:
        decoded_frames = []
        frame_converter = VideoReformatter()
        try:
            video = av.open(file, mode="r", format=container_format or self.container_format)
            try:
                for frame_i, raw_frame in enumerate(video.decode(video=0)):
                    frame_image = frame_converter.reformat(raw_frame, format="rgb24").to_ndarray()
                    if frame_rate is None or raw_frame.time is None:
                        # Without decoded timestamps, align frames by decode order
                        frame_index = frame_i
                    else:
                        # Convert decoded presentation time back to the intended input frame index
                        # This lets black-frame filling place surviving frames around dropped ones
                        frame_index = round(float(raw_frame.time) * float(frame_rate))
                    decoded_frames.append(
                        _DecodedVideoFrame(
                            image=frame_image.copy(),
                            frame_index=frame_index,
                        ),
                    )
            finally:
                video.close()
        except InvalidDataError as ex:
            raise RuntimeError("The encoded or damaged video stream could not be decoded.") from ex
        return decoded_frames

    def _get_frame_rate(self, *, frames: list[VideoFrame]) -> Fraction:
        if self.frame_rate is not None:
            return Fraction(str(self.frame_rate)).limit_denominator(1_000_000)

        if len(frames) < 2:
            # default to 30 fps
            return Fraction(numerator=30, denominator=1)

        # use median timestamp delta so a single irregular interval has less influence
        timestamps = np.array([frame.timestamp for frame in frames], dtype=np.float64)
        if not np.all(np.isfinite(timestamps)):
            raise ValueError("Frame timestamps must be finite when inferring frame_rate.")

        timestamp_deltas = np.diff(timestamps)
        if np.any(timestamp_deltas <= 0.0):
            raise ValueError("Frame timestamps must be strictly increasing when inferring frame_rate.")

        median_delta = float(np.median(timestamp_deltas))
        return Fraction(str(1.0 / median_delta)).limit_denominator(1_000_000)

    def _prepare_image_for_encoding(self, image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        image = self._convert_image(image=image)
        # yuv420p-style encoders need even dimensions, so we will pad here
        # Decoded frames are cropped back later
        pad_h = image.shape[0] % 2
        pad_w = image.shape[1] % 2
        if pad_h == 0 and pad_w == 0:
            return image

        return np.pad(
            image,
            pad_width=((0, pad_h), (0, pad_w), (0, 0)),
            mode="edge",
        )

    @override
    def get_config(self) -> dict[str, Any]:
        cfg = super().get_config()
        cfg["codec"] = self.codec
        cfg["container_format"] = self.container_format
        cfg["pixel_format"] = self.pixel_format
        cfg["frame_rate"] = self.frame_rate
        cfg["quantizer"] = self.quantizer
        cfg["crf"] = self.crf
        cfg["qp"] = self.qp
        cfg["bit_rate"] = self.bit_rate
        cfg["min_bit_rate"] = self.min_bit_rate
        cfg["max_bit_rate"] = self.max_bit_rate
        cfg["bit_rate_buffer_size"] = self.bit_rate_buffer_size
        cfg["gop_size"] = self.gop_size
        cfg["max_b_frames"] = self.max_b_frames
        cfg["encoder_options"] = dict(self.encoder_options)
        cfg["packet_loss_mode"] = self.packet_loss_mode
        cfg["packet_loss_model"] = self.packet_loss_model
        cfg["packet_loss_rate"] = self.packet_loss_rate
        cfg["packet_loss_burst_length"] = self.packet_loss_burst_length
        cfg["packet_loss_seed"] = self.packet_loss_seed
        cfg["packet_loss_preserve_keyframes"] = self.packet_loss_preserve_keyframes
        cfg["transport_loss_preserve_payload_starts"] = self.transport_loss_preserve_payload_starts
        cfg["gilbert_elliott_good_loss_rate"] = self.gilbert_elliott_good_loss_rate
        cfg["gilbert_elliott_bad_loss_rate"] = self.gilbert_elliott_bad_loss_rate
        cfg["gilbert_elliott_good_to_bad_rate"] = self.gilbert_elliott_good_to_bad_rate
        cfg["gilbert_elliott_bad_to_good_rate"] = self.gilbert_elliott_bad_to_good_rate
        cfg["frame_count_policy"] = self.frame_count_policy
        return cfg
