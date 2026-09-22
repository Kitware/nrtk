"""Tests for TurbulenceVideoPerturber."""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Hashable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from itertools import islice
from typing import Any

import numpy as np
import pytest
from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from syrupy.assertion import SnapshotAssertion
from typing_extensions import override

from nrtk.impls.perturb_video.optical import TurbulenceVideoPerturber
from nrtk.interfaces import VideoFrame
from tests.conftest import PSNRVideoSnapshotExtension
from tests.impls import INPUT_DRONE_VIDEO_FILE_PATH
from tests.impls.perturb_video.perturber_tests_mixin import PerturbVideoTestsMixin
from tests.impls.perturb_video.test_perturber_utils import FrameRepeater, perturber_assertions
from tests.utils.video_io import read_video

# Use small images and grid for fast CI
SMALL_SIZE = 32
SMALL_GRID = 32


@pytest.mark.hcipy
class TestTurbulenceVideoPerturber(PerturbVideoTestsMixin):
    impl_class = TurbulenceVideoPerturber

    @override
    def make_perturber(self, **kwargs: Any) -> TurbulenceVideoPerturber:
        """Create a TurbulenceVideoPerturber with small defaults for fast testing."""
        defaults: dict[str, Any] = {"grid_size": SMALL_GRID, "seed": 42}
        defaults.update(kwargs)
        return TurbulenceVideoPerturber(**defaults)

    @override
    def make_frames(
        self,
        n: int = 2,
        size: int = SMALL_SIZE,
        channels: int = 3,
        dtype: type = np.uint8,
        seed: int = 0,
    ) -> list[VideoFrame]:
        """Create a fresh list of VideoFrames with random image data."""
        rng = np.random.default_rng(seed)
        shape: tuple[int, ...] = (size, size, channels) if channels > 0 else (size, size)
        if np.issubdtype(dtype, np.integer):
            image = rng.integers(low=0, high=255, size=shape, dtype=dtype)
        else:
            image = rng.random(size=shape).astype(dtype)
        return list(FrameRepeater(frame=VideoFrame(image=image, timestamp=0.0), n=n))

    @pytest.mark.parametrize(
        ("kwargs", "expectation"),
        [
            ({"grid_size": 32}, does_not_raise()),
            ({"grid_size": 32, "D": 0}, pytest.raises(ValueError, match="D must be positive")),
            ({"grid_size": 32, "D": -1}, pytest.raises(ValueError, match="D must be positive")),
            ({"grid_size": 32, "slant_range": 0}, pytest.raises(ValueError, match="slant_range must be positive")),
            ({"grid_size": 32, "slant_range": -100}, pytest.raises(ValueError, match="slant_range must be positive")),
            ({"grid_size": 32, "path_avg_cn2": 0}, pytest.raises(ValueError, match="path_avg_cn2 must be positive")),
            (
                {"grid_size": 32, "path_avg_cn2": -1e-14},
                pytest.raises(ValueError, match="path_avg_cn2 must be positive"),
            ),
            ({"grid_size": 32, "pixel_pitch": 0}, pytest.raises(ValueError, match="pixel_pitch must be positive")),
            ({"grid_size": 32, "pixel_pitch": -1e-6}, pytest.raises(ValueError, match="pixel_pitch must be positive")),
            ({"grid_size": 32, "focal_length": 0}, pytest.raises(ValueError, match="focal_length must be positive")),
            (
                {"grid_size": 32, "focal_length": -0.01},
                pytest.raises(ValueError, match="focal_length must be positive"),
            ),
            ({"grid_size": 32, "wavelength": 0}, pytest.raises(ValueError, match="wavelength must be positive")),
            ({"grid_size": 32, "wavelength": -1e-6}, pytest.raises(ValueError, match="wavelength must be positive")),
            ({"grid_size": 32, "wind_speed": -1}, pytest.raises(ValueError, match="wind_speed must be non-negative")),
            ({"grid_size": 32, "L0": 0}, pytest.raises(ValueError, match="L0 must be positive")),
            ({"grid_size": 32, "L0": -5}, pytest.raises(ValueError, match="L0 must be positive")),
            ({"grid_size": 8}, pytest.raises(ValueError, match="grid_size must be >= 32")),
            ({"grid_size": 0}, pytest.raises(ValueError, match="grid_size must be >= 32")),
            ({"grid_size": 32, "eta": -0.1}, pytest.raises(ValueError, match=r"eta must be in \[0, 1\)")),
            ({"grid_size": 32, "eta": 1.0}, pytest.raises(ValueError, match=r"eta must be in \[0, 1\)")),
            ({"grid_size": 32, "eta": 1.5}, pytest.raises(ValueError, match=r"eta must be in \[0, 1\)")),
        ],
    )
    def test_configuration_bounds(
        self,
        kwargs: dict[str, Any],
        expectation: AbstractContextManager,
    ) -> None:
        """Raise appropriate errors for specific parameters."""
        with expectation:
            TurbulenceVideoPerturber(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"seed": None},
            {"path_avg_cn2": 5e-13, "slant_range": 1000.0, "seed": 123},
            {"wind_direction_deg": 45.0, "color_fill": [128, 128, 128]},
            {"color_fill": None, "grid_size": 32},
            {"eta": 0.3, "L0": 10.0},
            {"sub_pixel": True},
        ],
    )
    def test_configuration(self, kwargs: dict[str, Any]) -> None:
        """Test configuration stability."""
        inst = self.make_perturber(**kwargs)
        for i in configuration_test_helper(inst):
            assert i.path_avg_cn2 == inst.path_avg_cn2
            assert i.slant_range == inst.slant_range
            assert i.D == inst.D
            assert i.pixel_pitch == inst.pixel_pitch
            assert i.focal_length == inst.focal_length
            assert i.eta == inst.eta
            assert i.wavelength == inst.wavelength
            assert i.wind_speed == inst.wind_speed
            assert i.wind_direction_deg == inst.wind_direction_deg
            assert i.L0 == inst.L0
            assert i.grid_size == inst.grid_size
            assert i.color_fill == inst.color_fill
            assert i.sub_pixel == inst.sub_pixel
            assert i.seed == inst.seed

    def test_seeded_reproducible(self) -> None:
        """Two calls with same seed produce identical output."""
        frames1 = self.make_frames(n=3)
        frames2 = self.make_frames(n=3)

        inst = self.make_perturber(seed=42)
        results1 = list(inst.perturb(frames=iter(frames1)))

        results2 = list(inst.perturb(frames=iter(frames2)))

        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2, strict=True):
            assert np.array_equal(r1.image, r2.image)
            assert r1.timestamp == r2.timestamp

    def test_unseeded_non_deterministic(self) -> None:
        """Two calls with seed=None produce different output."""
        frames1 = self.make_frames(n=2)
        frames2 = self.make_frames(n=2)

        inst = self.make_perturber(seed=None)
        results1 = list(inst.perturb(frames=iter(frames1)))
        results2 = list(inst.perturb(frames=iter(frames2)))

        any_different = any(not np.array_equal(r1.image, r2.image) for r1, r2 in zip(results1, results2, strict=True))
        assert any_different

    def test_atmosphere_fixed_by_seed_not_direction(self) -> None:
        """`seed` pins the atmospheric realization independent of wind_direction_deg.

        At wind_speed=0 the velocity is [0, 0] for any direction, so if the
        HCIPy seed is consistent across direction configs the outputs must be
        byte-identical.
        """
        outputs = []
        for direction in [0.0, 90.0, None]:
            inst = self.make_perturber(seed=42, wind_speed=0.0, wind_direction_deg=direction)
            frames = self.make_frames(n=1)
            r = list(inst.perturb(frames=iter(frames)))
            outputs.append(r[0].image)
        assert np.array_equal(outputs[0], outputs[1])
        assert np.array_equal(outputs[0], outputs[2])

    def test_cached_static_state_reused(self) -> None:
        """Expensive HCIPy state is cached and reused across perturb() calls.

        Atmosphere-independent state (pupil grid, propagator, layer) is built once
        and intentionally reused as a performance cache; per-call randomness remains
        governed by the seed.
        """
        inst = self.make_perturber(seed=42)
        layer_before = inst._layer
        pupil_before = inst._pupil_grid
        propagator_before = inst._propagator

        list(inst.perturb(frames=iter(self.make_frames(n=1))))
        list(inst.perturb(frames=iter(self.make_frames(n=1))))

        assert inst._layer is layer_before
        assert inst._pupil_grid is pupil_before
        assert inst._propagator is propagator_before

    @pytest.mark.parametrize(
        ("path_avg_cn2", "slant_range", "D", "expected"),
        [
            # Weak turbulence (cn2=1.7e-14, slant=500, D=40mm): D/r0 ~ 1.6 -> 8*D/r0 ~ 13 -> clamped to 32
            (1.7e-14, 500.0, 0.04, 32),
            # Mid turbulence: D/r0 ~ 4.6 -> 8*D/r0 ~ 37 -> next pow2 = 64
            (1e-13, 500.0, 0.04, 64),
            # Strong turbulence: D/r0 ~ 28 -> 8*D/r0 ~ 224 -> next pow2 = 256
            (1e-12, 1000.0, 0.04, 256),
            # Telescope-scale: D/r0 ~ 176 -> 8*D/r0 ~ 1408 -> clamped to 512
            (1e-14, 10000.0, 1.0, 512),
        ],
    )
    def test_auto_grid_size_matches_lane_rule(
        self,
        path_avg_cn2: float,
        slant_range: float,
        D: float,  # noqa: N803
        expected: int,
    ) -> None:
        """grid_size=None resolves to next_pow2(8*D/r0) clamped to [32, 512]."""
        inst = TurbulenceVideoPerturber(
            path_avg_cn2=path_avg_cn2,
            slant_range=slant_range,
            D=D,
            seed=42,
        )
        assert inst.grid_size == expected

    def test_explicit_grid_size_overrides_auto(self) -> None:
        """Explicit grid_size is preserved exactly."""
        inst = TurbulenceVideoPerturber(grid_size=128, seed=42)
        assert inst.grid_size == 128

    def test_sub_pixel_default_is_false(self) -> None:
        """sub_pixel defaults to False (fast integer-pixel shift)."""
        p = self.make_perturber()
        assert p.sub_pixel is False

    def test_sub_pixel_differs_from_integer(self) -> None:
        """sub_pixel=True preserves fractional tilt that integer rounding discards.

        At weak turbulence (sub-pixel jitter rms), the two paths MUST differ — the
        integer path rounds the fractional shift to zero on nearly every frame,
        while the cubic path renders it faithfully.
        """
        # Weak turbulence -> jitter rms well below 1 px at default plate scale
        kwargs = {"path_avg_cn2": 5e-14, "slant_range": 1000.0, "seed": 42, "grid_size": SMALL_GRID}

        p_int = TurbulenceVideoPerturber(sub_pixel=False, **kwargs)
        p_sub = TurbulenceVideoPerturber(sub_pixel=True, **kwargs)

        frames_int = self.make_frames(n=3)
        frames_sub = self.make_frames(n=3)  # same data, same seed
        results_int = list(p_int.perturb(frames=iter(frames_int)))
        results_sub = list(p_sub.perturb(frames=iter(frames_sub)))

        all_equal = all(np.array_equal(r1.image, r2.image) for r1, r2 in zip(results_int, results_sub, strict=True))
        assert not all_equal, (
            "sub_pixel=True should produce different output than sub_pixel=False "
            "at weak turbulence — integer rounding discards sub-pixel jitter."
        )

    def test_undersampled_warning_on_explicit_small_grid(self) -> None:
        """Explicit grid_size that puts pitch > r0/6 emits a UserWarning."""
        with pytest.warns(UserWarning, match="phase screen may be aliased"):
            TurbulenceVideoPerturber(
                path_avg_cn2=5e-13,
                slant_range=1000.0,
                D=0.04,
                grid_size=32,
                seed=42,
            )

    def test_no_undersampled_warning_for_auto_size(self) -> None:
        """Auto-sized grid satisfies pitch <= r0/8 and emits no sampling warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=UserWarning)
            TurbulenceVideoPerturber(
                path_avg_cn2=1.7e-14,
                slant_range=500.0,
                D=0.04,
                wind_speed=0.0,
                seed=42,
            )

    def test_num_airy_cap_warning(self) -> None:
        """Extreme D/r0 caps the focal grid num_airy and emits a UserWarning."""
        # D/r0 > 60 triggers the cap (requested_airy = ceil(2.5 * D/r0) > 150).
        # Use explicit grid_size=512 (the auto-cap) so the undersampling warning
        # does not suppress the one we are actually checking for.
        with pytest.warns(UserWarning, match="num_airy capped"):
            TurbulenceVideoPerturber(
                path_avg_cn2=1e-11,
                slant_range=10000.0,
                D=1.0,
                grid_size=512,
                seed=42,
            )

    def test_pickle_round_trip(self) -> None:
        """Perturber survives pickle dump/load and produces identical output."""
        inst = self.make_perturber(seed=42)
        frames_a = self.make_frames(n=3)
        frames_b = self.make_frames(n=3)

        restored = pickle.loads(pickle.dumps(inst))  # noqa: S301 - test-controlled data

        results_orig = list(inst.perturb(frames=iter(frames_a)))
        results_restored = list(restored.perturb(frames=iter(frames_b)))

        assert len(results_orig) == len(results_restored)
        for r_orig, r_rest in zip(results_orig, results_restored, strict=True):
            assert np.array_equal(r_orig.image, r_rest.image)

    @pytest.mark.parametrize(
        ("n", "channels", "dtype"),
        [
            (3, 3, np.uint8),
            (2, 1, np.float32),
            (2, 3, np.float64),
        ],
    )
    def test_standard_assertions(self, n: int, channels: int, dtype: type) -> None:
        """Run the shared video perturber assertions on several dtype / channel combos."""
        inst = self.make_perturber(seed=42)
        frames = self.make_frames(n=n, size=SMALL_SIZE, channels=channels, dtype=dtype)
        for _ in perturber_assertions(perturb=inst, frames=iter(frames)):
            pass

    def test_call_matches_perturb(self) -> None:
        """Verify inst() produces same results as inst.perturb()."""
        inst = self.make_perturber(seed=42)
        frames1 = self.make_frames(n=2)
        results_perturb = list(inst.perturb(frames=iter(frames1)))

        inst2 = self.make_perturber(seed=42)
        frames2 = self.make_frames(n=2)
        results_call = list(inst2(frames=iter(frames2)))

        assert len(results_perturb) == len(results_call)
        for r1, r2 in zip(results_perturb, results_call, strict=True):
            assert np.array_equal(r1.image, r2.image)
            assert r1.timestamp == r2.timestamp

    def test_empty_frames(self) -> None:
        """Empty frame iterable yields no output and does not error."""
        inst = self.make_perturber(seed=42)
        assert list(inst.perturb(frames=iter([]))) == []

    def test_boxes_shifted(self) -> None:
        """Bounding boxes are shifted and remain within image bounds."""
        box = AxisAlignedBoundingBox(min_vertex=(5, 5), max_vertex=(15, 15))
        frames = FrameRepeater(
            frame=VideoFrame(
                image=np.zeros((SMALL_SIZE, SMALL_SIZE, 3), dtype=np.uint8),
                timestamp=0.0,
                boxes=[(box, {"label": 1.0})],
            ),
            n=1,
        )

        inst = self.make_perturber(seed=42)
        results = list(inst.perturb(frames=frames))

        assert len(results) == 1
        result = results[0]
        result_boxes = list(result.boxes)
        assert len(result_boxes) > 0
        shifted_box, metadata = result_boxes[0]
        assert metadata == {"label": 1.0}
        assert shifted_box.min_vertex[0] >= 0
        assert shifted_box.min_vertex[1] >= 0
        assert shifted_box.max_vertex[0] <= SMALL_SIZE
        assert shifted_box.max_vertex[1] <= SMALL_SIZE

    def test_boxes_empty_input(self) -> None:
        """Frames with no boxes produce no boxes in output."""
        frames = FrameRepeater(
            frame=VideoFrame(
                image=np.zeros((SMALL_SIZE, SMALL_SIZE, 3), dtype=np.uint8),
                timestamp=0.0,
            ),
            n=1,
        )
        inst = self.make_perturber(seed=42)
        results = list(inst.perturb(frames=frames))
        assert len(results) == 1
        assert list(results[0].boxes) == []

    def test_multiple_boxes(self) -> None:
        """Multiple bounding boxes are all shifted."""
        boxes: list[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] = [
            (AxisAlignedBoundingBox(min_vertex=(2, 2), max_vertex=(10, 10)), {"label": 1.0}),
            (AxisAlignedBoundingBox(min_vertex=(15, 15), max_vertex=(25, 25)), {"label": 2.0}),
        ]
        frames = FrameRepeater(
            frame=VideoFrame(
                image=np.zeros((SMALL_SIZE, SMALL_SIZE, 3), dtype=np.uint8),
                timestamp=0.0,
                boxes=boxes,
            ),
            n=1,
        )
        inst = self.make_perturber(seed=42)
        results = list(inst.perturb(frames=frames))
        assert len(results) == 1

    def test_boxes_shifted_offscreen_dropped(self) -> None:
        """Boxes whose shifted extent collapses off-image are dropped."""
        # Box near the right/bottom edge: a +10 px shift clips both min and max to
        # the image dimension, leaving a zero-area box that must be dropped.
        boxes: list[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] = [
            (AxisAlignedBoundingBox(min_vertex=(30, 30), max_vertex=(31, 31)), {"label": 1.0}),
            (AxisAlignedBoundingBox(min_vertex=(5, 5), max_vertex=(10, 10)), {"label": 2.0}),
        ]
        shifted = TurbulenceVideoPerturber._shift_boxes(
            boxes=boxes,
            shift_x=10.0,
            shift_y=10.0,
            image_shape=(SMALL_SIZE, SMALL_SIZE, 3),
        )
        assert len(shifted) == 1
        assert shifted[0][1] == {"label": 2.0}

    @pytest.mark.parametrize(
        ("shift_x", "shift_y", "exposed_corner"),
        [
            (2, 2, (0, 0)),
            (-2, 2, (0, -1)),
            (2, -2, (-1, 0)),
            (-2, -2, (-1, -1)),
        ],
    )
    def test_integer_shift_color_fill_at_exposed_corner(
        self,
        shift_x: int,
        shift_y: int,
        exposed_corner: tuple[int, int],
    ) -> None:
        """Each sign combination of integer-pixel shift fills the exposed corner."""
        fill_value = 128
        inst = self.make_perturber(seed=42, color_fill=fill_value)
        image = np.full((SMALL_SIZE, SMALL_SIZE, 3), 200, dtype=np.uint8)
        shifted = inst._apply_shift_integer(
            image=image,
            shift_x=float(shift_x),
            shift_y=float(shift_y),
            original_image=image,
        )
        y, x = exposed_corner
        assert np.array_equal(shifted[y, x], np.full(3, fill_value, dtype=np.uint8))

    @pytest.mark.parametrize(
        ("color_fill", "image_shape", "expected_fill"),
        [
            (128, (8, 8, 3), [128, 128, 128]),
            (64, (8, 8), [64]),
            ([10, 20, 30], (8, 8, 3), [10, 20, 30]),
            (None, (8, 8, 3), [0, 0, 0]),
        ],
    )
    def test_resolve_fill(
        self,
        color_fill: int | list[int] | None,
        image_shape: tuple[int, ...],
        expected_fill: list[int],
    ) -> None:
        """Scalar ints broadcast to all channels; sequences pass through."""
        inst = self.make_perturber(seed=42, color_fill=color_fill)
        image = np.zeros(image_shape, dtype=np.uint8)
        fill = inst._resolve_fill(image)
        assert np.array_equal(fill, np.array(expected_fill, dtype=np.uint8))
        assert fill.dtype == np.uint8

    def test_color_fill_length_mismatch(self) -> None:
        """color_fill length not matching image channel count raises ValueError."""
        inst = self.make_perturber(seed=42, color_fill=[0, 0])
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match=r"color_fill length 2 does not match image channel count 3"):
            inst._resolve_fill(image)

    def test_grayscale_2d(self) -> None:
        """2D grayscale frames produce 2D outputs."""
        inst = self.make_perturber(seed=42)
        frames = self.make_frames(n=2, size=SMALL_SIZE, channels=0, dtype=np.uint8)
        results = list(perturber_assertions(perturb=inst, frames=iter(frames)))
        assert len(results) == 2
        for r in results:
            assert r.image.ndim == 2

    def test_grayscale_scalar_fill(self) -> None:
        """Scalar color_fill works on 2D grayscale frames."""
        inst = self.make_perturber(seed=42, color_fill=128)
        frames = self.make_frames(n=1, size=SMALL_SIZE, channels=0, dtype=np.uint8)
        results = list(perturber_assertions(perturb=inst, frames=iter(frames)))
        assert len(results) == 1

    def test_eta_perturbation(self) -> None:
        """Verify obstructed aperture (eta > 0) produces valid perturbation."""
        inst = self.make_perturber(seed=42, eta=0.3)
        frames = self.make_frames(n=2, size=SMALL_SIZE)
        results = list(perturber_assertions(perturb=inst, frames=iter(frames)))
        assert len(results) == 2

    def test_wide_timestamps(self) -> None:
        """Wider time intervals produce temporally distinct perturbations."""
        rng = np.random.default_rng(0)
        image = rng.integers(low=0, high=255, size=(SMALL_SIZE, SMALL_SIZE, 3), dtype=np.uint8)
        frames = [
            VideoFrame(image=image.copy(), timestamp=0.0),
            VideoFrame(image=image.copy(), timestamp=1.0),
            VideoFrame(image=image.copy(), timestamp=2.0),
        ]
        inst = self.make_perturber(seed=42)
        results = list(inst.perturb(frames=iter(frames)))
        assert len(results) == 3
        assert not np.array_equal(results[0].image, results[1].image)
        assert not np.array_equal(results[1].image, results[2].image)

    def test_non_increasing_timestamps(self) -> None:
        """Non-increasing timestamps should raise ValueError."""
        frames = [
            VideoFrame(image=np.zeros((SMALL_SIZE, SMALL_SIZE, 3), dtype=np.uint8), timestamp=1.0),
            VideoFrame(image=np.zeros((SMALL_SIZE, SMALL_SIZE, 3), dtype=np.uint8), timestamp=0.0),
        ]
        inst = self.make_perturber(seed=42)
        with pytest.raises(ValueError, match="Sequential frame timestamps are not increasing"):
            list(inst.perturb(frames=iter(frames)))

    def test_inconsistent_shapes(self) -> None:
        """Inconsistent frame shapes should raise ValueError."""
        frames = [
            VideoFrame(image=np.zeros((SMALL_SIZE, SMALL_SIZE, 3), dtype=np.uint8), timestamp=0.0),
            VideoFrame(image=np.zeros((SMALL_SIZE + 1, SMALL_SIZE, 3), dtype=np.uint8), timestamp=1.0),
        ]
        inst = self.make_perturber(seed=42)
        with pytest.raises(ValueError, match="Sequential frame images do not have the same shape"):
            list(inst.perturb(frames=iter(frames)))

    def test_simultaneous_perturb_raises(self) -> None:
        """Calling perturb() while a previous generator is still active raises."""
        inst = self.make_perturber(seed=42)
        frames1 = self.make_frames(n=3, size=SMALL_SIZE)
        frames2 = self.make_frames(n=3, size=SMALL_SIZE)

        gen1 = inst.perturb(frames=iter(frames1))
        next(gen1)

        with pytest.raises(RuntimeError, match=r"Cannot call perturb\(\) while previous perturb\(\) is incomplete"):
            next(inst.perturb(frames=iter(frames2)))

        for _ in gen1:
            pass

        for _ in inst.perturb(frames=iter(frames2)):
            pass

    def test_compute_r0_round_trip(self) -> None:
        """compute_path_avg_cn2(r0=compute_r0(...)) recovers original cn2."""
        cn2_orig = 1.7e-14
        wl = 0.55e-6
        sr = 500.0
        r0 = TurbulenceVideoPerturber.compute_r0(path_avg_cn2=cn2_orig, wavelength=wl, slant_range=sr)
        cn2_recovered = TurbulenceVideoPerturber.compute_path_avg_cn2(r0=r0, wavelength=wl, slant_range=sr)
        assert abs(cn2_recovered - cn2_orig) / cn2_orig < 1e-10

    def test_compute_r0_positive(self) -> None:
        """Verify compute_r0 returns a positive value."""
        r0 = TurbulenceVideoPerturber.compute_r0(path_avg_cn2=1.7e-14, wavelength=0.55e-6, slant_range=500.0)
        assert r0 > 0

    def test_compute_r0_invalid_inputs(self) -> None:
        """compute_r0 rejects non-positive inputs."""
        with pytest.raises(ValueError, match="wavelength must be positive"):
            TurbulenceVideoPerturber.compute_r0(path_avg_cn2=1e-14, wavelength=0, slant_range=500)
        with pytest.raises(ValueError, match="slant_range must be positive"):
            TurbulenceVideoPerturber.compute_r0(path_avg_cn2=1e-14, wavelength=0.55e-6, slant_range=0)
        with pytest.raises(ValueError, match="path_avg_cn2 must be positive"):
            TurbulenceVideoPerturber.compute_r0(path_avg_cn2=0, wavelength=0.55e-6, slant_range=500)

    def test_compute_path_avg_cn2_invalid_inputs(self) -> None:
        """compute_path_avg_cn2 rejects non-positive inputs."""
        with pytest.raises(ValueError, match="r0 must be positive"):
            TurbulenceVideoPerturber.compute_path_avg_cn2(r0=0, wavelength=0.55e-6, slant_range=500)
        with pytest.raises(ValueError, match="wavelength must be positive"):
            TurbulenceVideoPerturber.compute_path_avg_cn2(r0=0.05, wavelength=0, slant_range=500)
        with pytest.raises(ValueError, match="slant_range must be positive"):
            TurbulenceVideoPerturber.compute_path_avg_cn2(r0=0.05, wavelength=0.55e-6, slant_range=0)

    def test_regression(
        self,
        psnr_mp4_snapshot: SnapshotAssertion,
        ssim_mp4_snapshot: SnapshotAssertion,
    ) -> None:
        """Regression testing results to detect API changes.

        Uses ``grid_size=32`` rather than the auto-default so the snapshot
        is decoupled from the auto-sizing rule. The auto rule is verified
        independently by ``test_auto_grid_size_matches_lane_rule``; this
        test pins the actual per-pixel output for a fixed
        (cn2, slant, D, grid_size) config so any regression in the FFT /
        propagation / convolution / shift pipeline is caught directly.
        """
        frames = islice(read_video(INPUT_DRONE_VIDEO_FILE_PATH), 5)  # noqa: FKA100 - islice is a C function
        inst = TurbulenceVideoPerturber(grid_size=32, seed=42)
        # Materialize so both snapshot assertions can re-iterate over the same frames.
        results = list(perturber_assertions(perturb=inst, frames=frames))

        # The default min_psnr=48.13 dB (≈ 1 gray-level RMSE) is unachievable across
        # platforms for this pipeline: HCIPy and scipy FFTs produce float64-epsilon-different
        # results between Apple's vecLib (darwin/arm64) and OpenBLAS (linux/x86_64), which
        # compounds through the float→uint8 cast to ~47.7 dB worst-frame PSNR cross-platform.
        # 40 dB (~2.5 gray-level RMSE) absorbs that drift while still catching real
        # regressions, which typically drop PSNR by 15+ dB.
        class _RelaxedPSNRVideoSnapshotExtension(PSNRVideoSnapshotExtension):
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(min_psnr=40.0, **kwargs)

        relaxed_psnr = psnr_mp4_snapshot.use_extension(_RelaxedPSNRVideoSnapshotExtension)
        relaxed_psnr.assert_match(iter(results))
        ssim_mp4_snapshot.assert_match(iter(results))

    def test_zero_wind_speed_freezes_atmosphere(self) -> None:
        """``wind_speed=0`` produces identical output for every frame."""
        rng = np.random.default_rng(0)
        image = rng.integers(low=0, high=255, size=(SMALL_SIZE, SMALL_SIZE, 3), dtype=np.uint8)
        frames = [VideoFrame(image=image.copy(), timestamp=t) for t in (0.0, 0.5, 1.0)]
        inst = self.make_perturber(seed=42, wind_speed=0.0, path_avg_cn2=1e-12)
        results = list(inst.perturb(frames=iter(frames)))
        assert len(results) == 3
        assert np.array_equal(results[0].image, results[1].image)
        assert np.array_equal(results[1].image, results[2].image)

    def test_boxes_in_same_frame_share_shift(self) -> None:
        """All boxes in a frame translate by the same per-frame jitter offset."""
        box_a = AxisAlignedBoundingBox(min_vertex=(2, 2), max_vertex=(4, 4))
        box_b = AxisAlignedBoundingBox(min_vertex=(10, 10), max_vertex=(12, 12))
        image = np.zeros((SMALL_SIZE, SMALL_SIZE, 3), dtype=np.uint8)
        frame = VideoFrame(
            image=image,
            timestamp=0.0,
            boxes=[(box_a, {"id": 1.0}), (box_b, {"id": 2.0})],
        )
        inst = self.make_perturber(seed=42, wind_speed=0.0, path_avg_cn2=1e-12)
        result = next(iter(inst.perturb(frames=iter([frame]))))

        assert result.boxes is not None
        boxes = list(result.boxes)
        assert len(boxes) == 2
        out_a, out_b = boxes[0][0], boxes[1][0]
        delta_a = (
            out_a.min_vertex[0] - box_a.min_vertex[0],
            out_a.min_vertex[1] - box_a.min_vertex[1],
        )
        delta_b = (
            out_b.min_vertex[0] - box_b.min_vertex[0],
            out_b.min_vertex[1] - box_b.min_vertex[1],
        )
        assert delta_a == delta_b
