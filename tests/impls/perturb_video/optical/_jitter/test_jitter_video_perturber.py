"""Tests for JitterVideoPerturber (construction, config, perturb loop, boxes)."""

from __future__ import annotations

import warnings
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from itertools import islice
from typing import Any

import numpy as np
import pytest
from scipy import signal
from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from syrupy.assertion import SnapshotAssertion
from typing_extensions import override

from nrtk.impls.perturb_video.optical import JitterVideoPerturber
from nrtk.impls.perturb_video.optical._jitter.los_motion import LOSMotionModel
from nrtk.interfaces import VideoFrame
from tests.impls import INPUT_DRONE_VIDEO_FILE_PATH
from tests.impls.perturb_video.perturber_tests_mixin import PerturbVideoTestsMixin
from tests.impls.perturb_video.test_perturber_utils import FrameRepeater, perturber_assertions
from tests.utils.video_io import read_video

# Use small images for fast CI
SMALL_SIZE = 32


@pytest.mark.scipy
class TestJitterVideoPerturber(PerturbVideoTestsMixin):
    impl_class = JitterVideoPerturber

    @override
    def make_perturber(self, **kwargs: Any) -> JitterVideoPerturber:
        """Create a JitterVideoPerturber with small, fast, visibly nonzero defaults."""
        defaults: dict[str, Any] = {
            "tones": [(20.0, 1.0)],
            "residual_rms": 2e-4,
            "pixel_pitch": 5e-6,
            "focal_length": 0.05,  # ifov = 100 urad/px -> ~2 px RMS: visibly nonzero, fast
            "max_duration_s": 5.0,
            "seed": 42,
        }
        defaults.update(kwargs)
        return JitterVideoPerturber(**defaults)

    @override
    def make_frames(
        self,
        n: int = 2,
        size: int = SMALL_SIZE,
        channels: int = 3,
        dtype: type = np.uint8,
        seed: int = 0,
    ) -> list[VideoFrame]:
        """Return a fresh list of VideoFrames valid for the perturber."""
        rng = np.random.default_rng(seed)
        shape: tuple[int, ...] = (size, size, channels) if channels > 0 else (size, size)
        if np.issubdtype(dtype, np.integer):
            image = rng.integers(low=0, high=255, size=shape, dtype=dtype)
        else:
            image = rng.random(size=shape).astype(dtype)
        return list(FrameRepeater(frame=VideoFrame(image=image, timestamp=0.0), n=n))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"seed": None},
            {"tones": [(20.0, 1.0), (5.0, 0.3)], "residual_rms": 12e-6, "roll_ratio": 0.5, "seed": 3},
            {"rolling_shutter_readout_s": 16e-3, "color_fill": [128, 128, 128]},
            {"sub_pixel": False, "hf_cutoff_hz": 40.0, "max_duration_s": 30.0},
            {"internal_fs": 4000.0, "integration_time": 8e-3},
        ],
    )
    def test_configuration(self, kwargs: dict[str, Any]) -> None:
        """Config round-trips through get_config/from_config (smqtk helper)."""
        defaults: dict[str, Any] = {"seed": 42}
        defaults.update(kwargs)
        inst = JitterVideoPerturber(**defaults)
        for i in configuration_test_helper(inst):
            assert i.tones == inst.tones
            assert i.broadband_beta == inst.broadband_beta
            assert i.broadband_weight == inst.broadband_weight
            assert i.residual_rms == inst.residual_rms
            assert i.severity == inst.severity
            assert i.roll_ratio == inst.roll_ratio
            assert i.pixel_pitch == inst.pixel_pitch
            assert i.focal_length == inst.focal_length
            assert i.integration_time == inst.integration_time
            assert i.rolling_shutter_readout_s == inst.rolling_shutter_readout_s
            assert i.sub_pixel == inst.sub_pixel
            assert i.color_fill == inst.color_fill
            assert i.internal_fs == inst.internal_fs
            assert i.hf_cutoff_hz == inst.hf_cutoff_hz
            assert i.max_duration_s == inst.max_duration_s
            assert i.seed == inst.seed

    def test_auto_internal_fs_is_capped(self) -> None:
        """The auto-derived trajectory sample rate is bounded at 20 kHz."""
        p = JitterVideoPerturber(integration_time=1e-4)  # 10/T_int would be 100 kHz
        assert p.internal_fs <= 20000.0

    @pytest.mark.parametrize(
        ("kwargs", "expectation"),
        [
            ({}, does_not_raise()),
            ({"broadband_beta": -0.5}, pytest.raises(ValueError, match="broadband_beta")),
            ({"broadband_weight": 1.5}, pytest.raises(ValueError, match=r"broadband_weight")),
            ({"broadband_weight": -0.1}, pytest.raises(ValueError, match=r"broadband_weight")),
            ({"broadband_weight": float("nan")}, pytest.raises(ValueError, match="broadband_weight")),
            ({"residual_rms": -1e-6}, pytest.raises(ValueError, match="residual_rms")),
            ({"residual_rms": float("nan")}, pytest.raises(ValueError, match="residual_rms")),
            ({"severity": -1.0}, pytest.raises(ValueError, match="severity")),
            ({"roll_ratio": -0.5}, pytest.raises(ValueError, match="roll_ratio")),
            ({"pixel_pitch": 0.0}, pytest.raises(ValueError, match="pixel_pitch")),
            ({"focal_length": 0.0}, pytest.raises(ValueError, match="focal_length")),
            ({"integration_time": 0.0}, pytest.raises(ValueError, match="integration_time")),
            ({"rolling_shutter_readout_s": 0.0}, pytest.raises(ValueError, match="rolling_shutter_readout_s")),
            ({"rolling_shutter_readout_s": float("nan")}, pytest.raises(ValueError, match="rolling_shutter_readout_s")),
            ({"internal_fs": -1.0}, pytest.raises(ValueError, match="internal_fs")),
            ({"hf_cutoff_hz": 0.0}, pytest.raises(ValueError, match="hf_cutoff_hz")),
            # At or above Nyquist of the trajectory rate, scipy's filter design would raise from
            # inside perturb(); reject at construction against the resolved internal_fs instead
            # (auto-derived internal_fs caps at 20 kHz, so 20 kHz is never below Nyquist).
            ({"hf_cutoff_hz": 500.0, "internal_fs": 1000.0}, pytest.raises(ValueError, match="half the trajectory")),
            ({"hf_cutoff_hz": 20000.0}, pytest.raises(ValueError, match="half the trajectory")),
            ({"hf_cutoff_hz": 499.0, "internal_fs": 1000.0}, does_not_raise()),
            ({"max_duration_s": 0.0}, pytest.raises(ValueError, match="max_duration_s")),
            ({"tones": [(-5.0, 1.0)]}, pytest.raises(ValueError, match="tone frequencies")),
            ({"tones": [(5.0, -1.0)]}, pytest.raises(ValueError, match="tone weights")),
            ({"tones": [(float("nan"), 1.0)]}, pytest.raises(ValueError, match="tone frequencies")),
            ({"tones": [(5.0, float("nan"))]}, pytest.raises(ValueError, match="tone weights")),
        ],
    )
    def test_validation(self, kwargs: dict[str, Any], expectation: AbstractContextManager) -> None:
        """Bad parameters raise immediately at construction.

        Without up-front validation, broadband_weight > 1 produces silent all-NaN frames
        (then NaN->uint8 undefined behavior) and focal_length=0 surfaces later as an
        unhelpful ZeroDivisionError. NaN satisfies no ordering comparison, so the checks
        are written as negated-positive assertions to reject it rather than let it pass.
        """
        with expectation:
            JitterVideoPerturber(**kwargs)

    @pytest.mark.parametrize(
        ("color_fill", "match"),
        [
            ([128, 128], "does not match image channel count"),
            ([128, float("nan"), 128], "color_fill values must be finite"),
            ([128, float("inf"), 128], "color_fill values must be finite"),
        ],
    )
    def test_color_fill_validation(self, color_fill: list[float], match: str) -> None:
        """A bad color_fill is rejected at perturb time, once the channel count is known.

        A non-finite fill would otherwise land in revealed edge pixels and cast to uint8
        as undefined values.
        """
        inst = self.make_perturber(color_fill=color_fill)
        with pytest.raises(ValueError, match=match):
            list(inst.perturb(frames=iter(self.make_frames(n=1))))

    @pytest.mark.parametrize("color_fill", [128, [10, 20, 30]])
    def test_color_fill_int_and_sequence_accepted(self, color_fill: int | list[int]) -> None:
        """A scalar or per-channel color_fill resolves and perturbs without error."""
        out = list(self.make_perturber(color_fill=color_fill).perturb(frames=iter(self.make_frames(n=1))))
        assert out[0].image.shape == (SMALL_SIZE, SMALL_SIZE, 3)

    @pytest.mark.parametrize("tones", [[], [(20.0, 0.0)]])
    def test_inert_tones_warn_that_floor_takes_all_energy(self, tones: list[tuple[float, float]]) -> None:
        """With no tone energy (empty or all-zero weights), broadband_weight is inert and warns."""
        with pytest.warns(UserWarning, match="broadband_weight has no effect"):
            JitterVideoPerturber(tones=tones, broadband_weight=0.1)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            JitterVideoPerturber(tones=tones, broadband_weight=1.0)  # explicit all-floor: no warning

    @pytest.mark.parametrize("tones", [[], [(20.0, 0.0)]])
    def test_no_motion_source_warns(self, tones: list[tuple[float, float]]) -> None:
        """With no tone energy and no floor the perturber warns and leaves frames and boxes unchanged."""
        with pytest.warns(UserWarning, match="no motion source") as caught:
            inst = JitterVideoPerturber(tones=tones, broadband_weight=0.0)
        assert len(caught) == 1
        assert "broadband_weight has no effect" not in str(caught[0].message)
        frames = self.make_frames(n=1)
        box = AxisAlignedBoundingBox(min_vertex=(1.3, 2.2), max_vertex=(5.6, 6.7))
        frames[0].boxes = [(box, {"score": 1.0})]
        out = next(inst.perturb(frames=iter(frames)))
        assert np.array_equal(out.image, frames[0].image)
        obox, _meta = next(iter(out.boxes))
        assert tuple(obox.min_vertex) == pytest.approx((1.3, 2.2))
        assert tuple(obox.max_vertex) == pytest.approx((5.6, 6.7))

    def test_perturb_yields_same_count_shape_dtype(self) -> None:
        """Output preserves frame count, shape, and dtype."""
        out = list(self.make_perturber().perturb(frames=iter(self.make_frames(n=10))))
        assert len(out) == 10
        assert out[0].image.shape == (SMALL_SIZE, SMALL_SIZE, 3)
        assert out[0].image.dtype == np.uint8

    def test_perturb_is_deterministic_with_seed(self) -> None:
        """Same seed and timestamps produce bit-identical output across calls."""
        a = [f.image for f in self.make_perturber().perturb(frames=iter(self.make_frames(n=5)))]
        b = [f.image for f in self.make_perturber().perturb(frames=iter(self.make_frames(n=5)))]
        for x, y in zip(a, b, strict=True):
            np.testing.assert_array_equal(x, y)

    def test_perturb_actually_changes_frames(self) -> None:
        """The perturbation visibly modifies frame content at the test amplitude."""
        frames = self.make_frames(n=5)
        out = list(self.make_perturber().perturb(frames=iter(self.make_frames(n=5))))
        assert not np.array_equal(out[2].image, frames[2].image)

    def test_aliasing_contract(self) -> None:
        """Repo-wide contract: inputs unmodified, no shared memory, dtype preserved."""
        frames = self.make_frames(n=3)
        for frame in frames:
            frame.boxes = [
                (AxisAlignedBoundingBox(min_vertex=(10.0, 10.0), max_vertex=(30.0, 30.0)), {"score": 1.0}),
            ]
            frame.additional_params = {"nested": {"k": [1, 2]}}
        inst = self.make_perturber()
        for _ in perturber_assertions(perturb=inst.perturb, frames=iter(frames)):
            pass

    def test_box_contains_object_under_roll_and_shift(self) -> None:
        """After roll + displacement, the shifted box must still contain the warped object.

        A rotation-direction mismatch between the image warp and the box transform would
        silently corrupt ground truth; this pins the two to a single convention.
        """
        img = np.zeros((101, 101, 3), dtype=np.uint8)
        img[28:33, 78:83] = 255  # bright 5x5 patch right of center
        box = AxisAlignedBoundingBox(min_vertex=(76.0, 26.0), max_vertex=(85.0, 35.0))
        frames = [VideoFrame(image=img, timestamp=0.0, boxes=[(box, {"score": 1.0})])]
        inst = self.make_perturber(
            tones=[(5.0, 1.0)],
            residual_rms=3e-3,
            roll_ratio=1.0,
            pixel_pitch=5e-6,
            focal_length=0.005,
        )  # wide FOV (1 mrad/px): a few px shift + visible rotation
        out = next(inst.perturb(frames=iter(frames)))
        assert out.boxes, "box was dropped"
        obox, _meta = next(iter(out.boxes))
        row, col = np.unravel_index(int(np.argmax(out.image[..., 0])), out.image[..., 0].shape)
        assert obox.min_vertex[0] - 1 <= col <= obox.max_vertex[0] + 1
        assert obox.min_vertex[1] - 1 <= row <= obox.max_vertex[1] + 1

    def test_rolling_shutter_produces_row_dependent_shift(self) -> None:
        """Rolling shutter: a tone with period == readout time bends a vertical line sinusoidally."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, 50] = 255  # vertical line
        frames = [VideoFrame(image=img, timestamp=0.0)]
        inst = self.make_perturber(
            tones=[(62.5, 1.0)],  # period 16 ms == readout -> one full wiggle down the frame
            broadband_weight=0.0,
            residual_rms=5e-4,
            rolling_shutter_readout_s=16e-3,
            integration_time=1e-4,  # near-instant rows: pure displacement
            internal_fs=8000.0,
        )
        out = next(inst.perturb(frames=iter(frames))).image[..., 0]
        line_cols = np.array([np.argmax(out[r]) for r in range(10, 90)], dtype=np.float64)
        assert line_cols.max() - line_cols.min() >= 2.0  # rows visibly disagree (jello)

    def test_box_contains_bent_object_under_rolling_shutter(self) -> None:
        """A tall object bent by rolling shutter stays inside its box, including between the corners.

        Each row carries its own displacement, so the object's extrema fall on rows strictly
        between the box corners; mapping the corners alone returns a box the object leaves.
        Smear is disabled so the check is about geometry, not blur.
        """
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[10:91, 50] = 255  # tall, thin vertical object
        box = AxisAlignedBoundingBox(min_vertex=(48.0, 8.0), max_vertex=(52.0, 92.0))
        frames = [VideoFrame(image=img, timestamp=0.0, boxes=[(box, {"score": 1.0})])]
        inst = self.make_perturber(
            tones=[(62.5, 1.0)],  # period 16 ms == readout -> one full bend down the frame
            broadband_weight=0.0,
            residual_rms=5e-4,  # ~7 px tone amplitude at 100 urad/px
            rolling_shutter_readout_s=16e-3,
            integration_time=1e-4,  # near-instant rows: no smear
            internal_fs=8000.0,
            roll_ratio=0.0,
        )
        out = next(inst.perturb(frames=iter(frames)))
        assert out.boxes, "box was dropped"
        obox, _meta = next(iter(out.boxes))
        rows, cols = np.nonzero(out.image[..., 0] > 100)
        assert obox.min_vertex[0] <= cols.min()
        assert cols.max() <= obox.max_vertex[0]
        assert obox.min_vertex[1] <= rows.min()
        assert rows.max() <= obox.max_vertex[1]
        assert cols.max() - cols.min() >= 10  # the bend is real: far wider than the 4 px input box

    def test_box_contains_object_under_folded_readout(self) -> None:
        """Strong rolling-shutter motion folds the readout and the box covers every copy of the object.

        Several output rows read the same source row, so the object appears more than once
        down the frame.
        """
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[10:91, 50] = 255
        box = AxisAlignedBoundingBox(min_vertex=(48.0, 8.0), max_vertex=(52.0, 92.0))
        frames = [VideoFrame(image=img, timestamp=0.0, boxes=[(box, {"score": 1.0})])]
        inst = self.make_perturber(
            tones=[(62.5, 1.0)],  # one period down the frame
            broadband_weight=0.0,
            residual_rms=3e-3,  # 30 px RMS at 100 urad/px, up to ~2.7 px of change per row
            rolling_shutter_readout_s=16e-3,
            integration_time=1e-6,
            internal_fs=8000.0,
            roll_ratio=0.0,
            seed=42,
        )
        out = next(inst.perturb(frames=iter(frames)))
        assert out.boxes, "box was dropped"
        obox, _meta = next(iter(out.boxes))
        rows, cols = np.nonzero(out.image[..., 0] > 100)
        assert obox.min_vertex[0] <= cols.min()
        assert cols.max() <= obox.max_vertex[0]
        assert obox.min_vertex[1] <= rows.min()
        assert rows.max() <= obox.max_vertex[1]

    def test_normalized_float_frames_stay_in_unit_range(self) -> None:
        """Cubic resampling overshoots at hard edges; normalized float frames are kept in [0, 1]."""
        img = np.zeros((SMALL_SIZE, SMALL_SIZE, 3), dtype=np.float32)
        img[:, SMALL_SIZE // 2 :] = 1.0  # a full-contrast vertical edge
        out = next(self.make_perturber().perturb(frames=iter([VideoFrame(image=img, timestamp=0.0)])))
        assert out.image.dtype == np.float32
        assert out.image.min() >= 0.0
        assert out.image.max() <= 1.0

    def test_unnormalized_float_frames_keep_their_range(self) -> None:
        """Float frames that are not normalized to [0, 1] are not clipped to it."""
        img = np.full((SMALL_SIZE, SMALL_SIZE, 3), 200.0, dtype=np.float32)
        out = next(self.make_perturber().perturb(frames=iter([VideoFrame(image=img, timestamp=0.0)])))
        assert out.image.max() > 100.0

    def test_additional_params_emission(self) -> None:
        """Each output frame reports its achieved degradation without touching input params."""
        frames = self.make_frames(n=2)
        frames[0].additional_params = {"upstream": 1}
        out = list(self.make_perturber().perturb(frames=iter(frames)))
        rec = out[0].additional_params["jitter"]
        assert set(rec) == {"dx_px", "dy_px", "roll_rad", "smear_extent_px", "rms_commanded_px"}
        assert out[0].additional_params["upstream"] == 1
        assert frames[0].additional_params == {"upstream": 1}  # input untouched

    def test_timestamp_beyond_max_duration_raises(self) -> None:
        """Timestamps past the trajectory buffer raise instead of freezing the motion."""
        inst = self.make_perturber(max_duration_s=1.0)
        frames = [VideoFrame(image=self.make_frames(n=1)[0].image, timestamp=2.0)]
        with pytest.raises(ValueError, match="max_duration_s"):
            list(inst.perturb(frames=iter(frames)))

    def test_negative_timestamp_raises(self) -> None:
        """Negative timestamps raise instead of sampling before the trajectory buffer."""
        inst = self.make_perturber()
        frames = [VideoFrame(image=self.make_frames(n=1)[0].image, timestamp=-0.5)]
        with pytest.raises(ValueError, match="non-negative"):
            list(inst.perturb(frames=iter(frames)))

    def test_box_fully_clipped_out_of_frame_is_dropped(self) -> None:
        """A box pushed entirely outside the frame is dropped, not clipped to a sliver."""
        box = AxisAlignedBoundingBox(min_vertex=(10.0, 10.0), max_vertex=(20.0, 20.0))
        inst = self.make_perturber()
        h = SMALL_SIZE
        out = inst._shift_frame_boxes(
            boxes=[(box, {"score": 1.0})],
            dx_rows=np.full(h, 2.0 * h),  # displacement larger than the frame
            dy_rows=np.zeros(h),
            roll_rad=0.0,
            image_shape=(h, h, 3),
        )
        assert out == []

    def test_zero_motion_is_identity_but_fresh_arrays(self) -> None:
        """severity=0 leaves pixels untouched while still returning new arrays."""
        frames = self.make_frames(n=2)
        out = list(self.make_perturber(severity=0.0).perturb(frames=iter(frames)))
        np.testing.assert_array_equal(out[0].image, frames[0].image)  # no 1-LSB warp residue
        assert not np.shares_memory(out[0].image, frames[0].image)

    def test_zero_motion_still_validates_timestamps(self) -> None:
        """severity=0 applies the same timestamp bounds as the perturbing path."""
        inst = self.make_perturber(severity=0.0, max_duration_s=1.0)
        frames = [VideoFrame(image=self.make_frames(n=1)[0].image, timestamp=2.0)]
        with pytest.raises(ValueError, match="max_duration_s"):
            list(inst.perturb(frames=iter(frames)))

    def test_zero_motion_still_validates_color_fill(self) -> None:
        """severity=0 applies the same color_fill validation as the perturbing path."""
        inst = self.make_perturber(severity=0.0, color_fill=[128, 128])
        with pytest.raises(ValueError, match="does not match image channel count"):
            list(inst.perturb(frames=iter(self.make_frames(n=1))))

    def test_zero_motion_emits_zeroed_record(self) -> None:
        """severity=0 frames still carry the degradation record, with all-zero values."""
        out = list(self.make_perturber(severity=0.0).perturb(frames=iter(self.make_frames(n=1))))
        rec = out[0].additional_params["jitter"]
        assert set(rec) == {"dx_px", "dy_px", "roll_rad", "smear_extent_px", "rms_commanded_px"}
        assert all(v == 0.0 for v in rec.values())

    def test_severity_scales_delivered_motion_exactly(self) -> None:
        """At a fixed seed, severity multiplies the delivered per-frame motion exactly.

        The same seed draws identical tone phases and floor; only the RMS normalization
        target changes, so severity=2 must deliver exactly twice the displacement of
        severity=1 frame-by-frame — severity is the sweep knob, not just a stored value.
        """

        def dx(severity: float) -> np.ndarray:
            out = self.make_perturber(severity=severity).perturb(frames=iter(self.make_frames(n=4)))
            return np.array([f.additional_params["jitter"]["dx_px"] for f in out])

        base = dx(1.0)
        assert np.abs(base).max() > 0.1  # motion is visibly nonzero at the test amplitude
        np.testing.assert_allclose(dx(2.0), 2.0 * base, rtol=1e-9)

    def test_grayscale_frames_supported(self) -> None:
        """Two-dimensional (grayscale) frames pass through the full pipeline."""
        out = list(self.make_perturber().perturb(frames=iter(self.make_frames(n=2, channels=0))))
        assert out[0].image.shape == (SMALL_SIZE, SMALL_SIZE)

    def test_regression(
        self,
        psnr_mp4_snapshot: SnapshotAssertion,
        ssim_mp4_snapshot: SnapshotAssertion,
    ) -> None:
        """Pin the rendered output for a fixed config within PSNR/SSIM tolerance.

        Complements (does not replace) the physics gates: PSNR stays green
        when the motion spectrum is wrong; phase-correlation stays green when rendering quality
        regresses.
        """
        # bitexact: the default SIMD YUV->RGB decode differs by a gray level between CPU architectures
        frames = islice(read_video(INPUT_DRONE_VIDEO_FILE_PATH, bitexact=True), 5)  # noqa: FKA100 - islice is a C function
        inst = JitterVideoPerturber.from_platform("helicopter", seed=42, max_duration_s=5.0)
        results = list(inst.perturb(frames=frames))

        psnr_mp4_snapshot.assert_match(iter(results))
        ssim_mp4_snapshot.assert_match(iter(results))


def _phase_corr_shift(*, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Integer phase-correlation shift of b relative to a, as (dx, dy)."""
    fa, fb = np.fft.rfft2(a), np.fft.rfft2(b)
    r = fb * np.conj(fa)
    r /= np.abs(r) + 1e-12
    corr = np.fft.irfft2(r, s=a.shape)
    dy, dx = np.unravel_index(int(np.argmax(corr)), corr.shape)
    if dy > a.shape[0] // 2:
        dy -= a.shape[0]
    if dx > a.shape[1] // 2:
        dx -= a.shape[1]
    return float(dx), float(dy)


def _textured(*, h: int = 128, w: int = 128) -> np.ndarray:
    return np.random.default_rng(1).integers(low=0, high=255, size=(h, w), dtype=np.uint8)


@pytest.mark.scipy
def test_phase_correlation_recovers_commanded_displacement() -> None:
    """End-to-end: measured image motion matches the emitted commanded motion.

    Phase-correlates each output frame against the input and compares the recovered
    displacement, its RMS, and its dominant spectral peak to the per-frame values the
    perturber reports about itself (integer-pixel precision).
    """
    base = _textured()
    n, fps = 100, 25.0
    frames = [VideoFrame(image=base.copy(), timestamp=i / fps) for i in range(n)]
    p = JitterVideoPerturber(
        tones=[(4.5, 1.0)],  # sub-Nyquist at 25 fps: no aliasing ambiguity
        broadband_weight=0.05,
        residual_rms=4e-4,
        pixel_pitch=5e-6,
        focal_length=0.05,  # ifov 100 urad/px -> commanded RMS ~4 px
        integration_time=1e-3,
        roll_ratio=0.0,
        max_duration_s=10.0,
        seed=7,
    )
    rec_dx, cmd_dx = [], []
    for out in p.perturb(frames=iter(frames)):
        dx, _dy = _phase_corr_shift(a=base.astype(np.float64), b=out.image.astype(np.float64))
        rec_dx.append(dx)
        cmd_dx.append(out.additional_params["jitter"]["dx_px"])
    rec, cmd = np.array(rec_dx), np.array(cmd_dx)
    assert np.abs(rec - cmd).mean() <= 1.0  # integer recovery tracks command frame-by-frame
    assert np.sqrt((rec**2).mean()) == pytest.approx(np.sqrt((cmd**2).mean()), rel=0.20)
    f, pxx = signal.welch(rec, fs=fps, nperseg=64)
    assert abs(f[np.argmax(pxx[1:]) + 1] - 4.5) <= fps / 64 + 1e-9  # dominant peak at the tone


@pytest.mark.scipy
def test_sinc_null_pure_tone_at_integer_ft_is_all_smear() -> None:
    """A tone with f*T_int = 1 has ZERO displacement (sinc null) — all its energy is smear.

    The midpoint-rule window mean makes the null exact; an endpoint-inclusive mean leaks
    ~1/n (~9%) of a pure-smear tone into displacement.
    """

    def dx_rms(t_int: float) -> float:
        frames = [VideoFrame(image=_textured(h=32, w=32), timestamp=i / 25.0) for i in range(40)]
        p = JitterVideoPerturber(
            tones=[(50.0, 1.0)],
            broadband_weight=0.0,
            residual_rms=4e-4,
            pixel_pitch=5e-6,
            focal_length=0.05,
            integration_time=t_int,
            roll_ratio=0.0,
            internal_fs=4000.0,
            max_duration_s=5.0,
            seed=3,
        )
        vals = [f.additional_params["jitter"]["dx_px"] for f in p.perturb(frames=iter(frames))]
        return float(np.sqrt(np.mean(np.array(vals) ** 2)))

    assert dx_rms(1.0 / 50.0) < 0.05 * dx_rms(1e-3)  # f*T=1 null vs f*T=0.05 pass-through


@pytest.mark.scipy
def test_impulse_image_yields_predicted_smear_kernel() -> None:
    """On a point source the output IS the smear kernel.

    Its second moment must match the within-exposure residual variance predicted by the
    motion model, verifying the smear channel end-to-end.
    """
    img = np.zeros((64, 64), dtype=np.float32)
    img[32, 32] = 1.0
    p = JitterVideoPerturber(
        tones=[(200.0, 1.0)],
        broadband_weight=0.0,
        residual_rms=3e-4,
        pixel_pitch=5e-6,
        focal_length=0.05,
        integration_time=10e-3,  # 2 full cycles: smear-dominated
        roll_ratio=0.0,
        max_duration_s=2.0,
        seed=11,
    )
    out = next(p.perturb(frames=iter([VideoFrame(image=img, timestamp=0.0)]))).image.astype(np.float64)
    # Predict from an identically-configured motion model, seeded the way perturb()
    # documents (re-seeded per call from the set seed):
    model = LOSMotionModel(
        tones=p.tones,
        broadband_beta=p.broadband_beta,
        broadband_weight=p.broadband_weight,
        residual_rms_rad=p.residual_rms * p.severity,
        roll_ratio=p.roll_ratio,
        internal_fs=p.internal_fs,
        hf_cutoff_hz=p.hf_cutoff_hz,
        max_duration_s=p.max_duration_s,
    )
    model.reset(rng=np.random.default_rng(11))
    t = (np.arange(256) + 0.5) * (10e-3 / 256)
    tx, ty, _ = model.sample(t=t)
    predicted_var = (
        float(((tx - tx.mean()) ** 2 + (ty - ty.mean()) ** 2).mean()) / (p.pixel_pitch / p.focal_length) ** 2
    )
    ys, xs = np.mgrid[0:64, 0:64]
    total = out.sum()
    cx, cy = (out * xs).sum() / total, (out * ys).sum() / total
    measured_var = float((out * ((xs - cx) ** 2 + (ys - cy) ** 2)).sum() / total)
    assert measured_var == pytest.approx(predicted_var, rel=0.5)  # loose: adds spline/pixelization blur


@pytest.mark.scipy
def test_longer_integration_time_increases_smear() -> None:
    """For a fixed spectrum, a longer exposure moves more energy into the smear channel."""

    def smear_extent(t_int: float) -> float:
        frames = [VideoFrame(image=_textured(h=32, w=32), timestamp=i / 25.0) for i in range(10)]
        p = JitterVideoPerturber(
            tones=[(100.0, 1.0)],
            residual_rms=3e-4,
            pixel_pitch=5e-6,
            focal_length=0.05,
            integration_time=t_int,
            roll_ratio=0.0,
            max_duration_s=2.0,
            seed=5,
        )
        vals = [f.additional_params["jitter"]["smear_extent_px"] for f in p.perturb(frames=iter(frames))]
        return float(np.mean(vals))

    assert smear_extent(20e-3) > smear_extent(2e-3)
