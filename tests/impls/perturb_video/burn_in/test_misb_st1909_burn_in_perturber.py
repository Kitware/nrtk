"""Tests for MISBST1909BurnInPerturber."""

from __future__ import annotations

import io
import time
from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import ImageFont
from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from syrupy.assertion import SnapshotAssertion

from nrtk.impls.perturb_video.burn_in import MISBST1909BurnInPerturber, _misb_st1909_burn_in_perturber
from nrtk.impls.perturb_video.burn_in._misb_st1909_burn_in_perturber import (
    _FONT_CACHE_SIZE,
    MISBST1909Metadata,
)
from nrtk.interfaces import VideoFrame
from tests.impls import INPUT_DRONE_VIDEO_FILE_PATH
from tests.impls.perturb_video.perturber_tests_mixin import PerturbVideoTestsMixin
from tests.impls.perturb_video.test_perturber_utils import perturber_assertions
from tests.utils.video_io import read_video

# Small frames keep CI fast while staying large enough for the text to render legibly.
SMALL_HEIGHT = 128
SMALL_WIDTH = 192

FIXED_METADATA: MISBST1909Metadata = {
    "main_sensor": {"name": "EO Nose", "azimuth_degrees": 123.4567, "elevation_degrees": -12.5},
    "classification_releasability": "UNCLASSIFIED//REL TO USA",
    "platform": {
        "name": "Platform 1",
        "latitude_degrees": 39.1234,
        "longitude_degrees": -77.9876,
        "altitude_meters": 3048.0,
    },
    "north_angle_degrees": 42.0,
    "next_zoom_box": AxisAlignedBoundingBox(min_vertex=(0.3, 0.3), max_vertex=(0.7, 0.65)),
    "laser_sensor": {"name": "Laser", "status": True, "prf_code": "1234"},
    "date_time": {
        "frame_date_time": datetime(year=2026, month=7, day=30, hour=12, minute=0, second=0),
        "metadata_date_time": datetime(year=2026, month=7, day=30, hour=12, minute=0, second=1),
    },
    "target": {
        "slant_range_meters": 5400.0,
        "target_width_meters": 120.0,
        "hfov_degrees": 3.5,
        "vfov_degrees": 2.1,
        "target_latitude_degrees": 39.0,
        "target_longitude_degrees": -77.0,
        "target_altitude_meters": 210.0,
    },
    "reticle_position": (0.5, 0.5),
}


@pytest.fixture(params=["UTC", "America/New_York", "Asia/Tokyo"])
def host_timezone(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Generator[str, None, None]:
    monkeypatch.setenv(name="TZ", value=request.param)
    time.tzset()
    yield request.param
    monkeypatch.undo()
    time.tzset()


@pytest.mark.pillow
class TestMISBST1909BurnInPerturber(PerturbVideoTestsMixin):
    impl_class = MISBST1909BurnInPerturber

    def make_perturber(self, **kwargs: Any) -> MISBST1909BurnInPerturber:
        """Create a MISBST1909BurnInPerturber with defaults suitable for fast testing."""
        defaults: dict[str, Any] = {"seed": 42}
        defaults.update(kwargs)
        return MISBST1909BurnInPerturber(**defaults)

    def make_frames(
        self,
        n: int = 2,
        height: int = SMALL_HEIGHT,
        width: int = SMALL_WIDTH,
        channels: int = 3,
        metadata: MISBST1909Metadata | None = None,
    ) -> list[VideoFrame]:
        """Create a fresh list of blank VideoFrames.

        The frames are black so that any non-zero pixel in the output is burn-in, which
        keeps the coverage-based assertions below unambiguous.
        """
        shape: tuple[int, ...] = (height, width, channels) if channels > 0 else (height, width)
        additional_params: dict[str, Any] = {"misb_st1909": metadata} if metadata is not None else {}
        return [
            VideoFrame(
                image=np.zeros(shape, dtype=np.uint8),
                timestamp=float(i),
                boxes=[],
                additional_params=dict(additional_params),
            )
            for i in range(n)
        ]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"seed": None},
            {"text_size": 1.0 / 24.0},
            {"color": [0, 255, 0], "outline_color": None},
            {"generate_missing_metadata": False},
            {"regenerate_metadata": False},
            {"zoom_box_corner_length": 0.25, "line_thickness": 1},
            {"reticle_outer_size": 0.2, "reticle_inner_size": 0.05, "north_circle_radius": 0.08},
        ],
    )
    def test_configuration(self, kwargs: dict[str, Any]) -> None:
        """Test configuration stability."""
        inst = self.make_perturber(**kwargs)
        for i in configuration_test_helper(inst):
            assert i.generate_missing_metadata == inst.generate_missing_metadata
            assert i.regenerate_metadata == inst.regenerate_metadata
            assert i.font_path == inst.font_path
            assert i.text_size == inst.text_size
            assert i.color == inst.color
            assert i.outline_color == inst.outline_color
            assert i.zoom_box_corner_length == inst.zoom_box_corner_length
            assert i.reticle_outer_size == inst.reticle_outer_size
            assert i.reticle_inner_size == inst.reticle_inner_size
            assert i.line_thickness == inst.line_thickness
            assert i.north_circle_radius == inst.north_circle_radius
            assert i.seed == inst.seed

    @pytest.mark.parametrize("channels", [3, 0])
    def test_standard_assertions(self, channels: int) -> None:
        """Run the shared video perturber assertions for color and grayscale input."""
        inst = self.make_perturber()
        frames = self.make_frames(n=3, channels=channels, metadata=FIXED_METADATA)
        for _ in perturber_assertions(perturb=inst, frames=iter(frames)):
            pass

    @pytest.mark.parametrize("channels", [3, 0])
    def test_output_is_rgb_uint8(self, channels: int) -> None:
        """Grayscale and color input alike are returned as three-channel uint8."""
        inst = self.make_perturber()
        results = list(inst.perturb(frames=iter(self.make_frames(n=1, channels=channels))))

        assert len(results) == 1
        assert results[0].image.shape == (SMALL_HEIGHT, SMALL_WIDTH, 3)
        assert results[0].image.dtype == np.uint8

    def test_frame_count_and_timestamps_preserved(self) -> None:
        """One frame out per frame in, with timestamps carried through."""
        frames = self.make_frames(n=4)
        results = list(self.make_perturber().perturb(frames=iter(frames)))

        assert len(results) == len(frames)
        assert [r.timestamp for r in results] == [f.timestamp for f in frames]

    def test_burn_ins_are_drawn(self) -> None:
        """The overlay actually marks the frame."""
        frames = self.make_frames(n=1, metadata=FIXED_METADATA)
        result = list(self.make_perturber().perturb(frames=iter(frames)))[0]

        assert result.image.any(), "expected burn-ins to be drawn over the blank frame"

    def test_empty_frames(self) -> None:
        """Empty frame iterable yields no output and does not error."""
        assert list(self.make_perturber().perturb(frames=iter([]))) == []

    def test_call_matches_perturb(self) -> None:
        """Verify inst() produces same results as inst.perturb()."""
        results_perturb = list(self.make_perturber().perturb(frames=iter(self.make_frames(n=2))))
        results_call = list(self.make_perturber()(frames=iter(self.make_frames(n=2))))

        assert len(results_perturb) == len(results_call)
        for r1, r2 in zip(results_perturb, results_call, strict=True):
            assert np.array_equal(r1.image, r2.image)
            assert r1.timestamp == r2.timestamp

    def test_boxes_and_params_passed_through(self) -> None:
        """A burn-in occludes pixels but never moves content, so annotations pass through."""
        box = AxisAlignedBoundingBox(min_vertex=(5, 5), max_vertex=(15, 15))
        frame = VideoFrame(
            image=np.zeros((SMALL_HEIGHT, SMALL_WIDTH, 3), dtype=np.uint8),
            timestamp=0.0,
            boxes=[(box, {"label": 1.0})],
            additional_params={"misb_st1909": FIXED_METADATA, "unrelated": "value"},
        )

        result = list(self.make_perturber().perturb(frames=iter([frame])))[0]

        result_boxes = list(result.boxes)
        assert len(result_boxes) == 1
        out_box, out_metadata = result_boxes[0]
        assert np.array_equal(out_box.min_vertex, box.min_vertex)
        assert np.array_equal(out_box.max_vertex, box.max_vertex)
        assert out_metadata == {"label": 1.0}
        assert result.additional_params["unrelated"] == "value"
        assert result.additional_params["misb_st1909"] == FIXED_METADATA

    def test_seeded_reproducible(self) -> None:
        """Two instances with the same seed generate the same metadata, so output matches."""
        results1 = list(self.make_perturber(seed=42).perturb(frames=iter(self.make_frames(n=2))))
        results2 = list(self.make_perturber(seed=42).perturb(frames=iter(self.make_frames(n=2))))

        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2, strict=True):
            assert np.array_equal(r1.image, r2.image)

    def test_unseeded_non_deterministic(self) -> None:
        """Two instances with seed=None generate different metadata."""
        results1 = list(self.make_perturber(seed=None).perturb(frames=iter(self.make_frames(n=2))))
        results2 = list(self.make_perturber(seed=None).perturb(frames=iter(self.make_frames(n=2))))

        assert any(not np.array_equal(r1.image, r2.image) for r1, r2 in zip(results1, results2, strict=True))

    def test_different_seeds_differ(self) -> None:
        """Different seeds generate different metadata."""
        results1 = list(self.make_perturber(seed=1).perturb(frames=iter(self.make_frames(n=1))))
        results2 = list(self.make_perturber(seed=2).perturb(frames=iter(self.make_frames(n=1))))

        assert not np.array_equal(results1[0].image, results2[0].image)

    def test_generated_metadata_is_static(self) -> None:
        """One set of random values is drawn per call and reused for every frame."""
        results = list(self.make_perturber(regenerate_metadata=False).perturb(frames=iter(self.make_frames(n=3))))

        for result in results[1:]:
            assert np.array_equal(results[0].image, result.image)

    def test_regenerate_metadata_varies_per_frame(self) -> None:
        """``regenerate_metadata`` redraws the random values for each frame."""
        results = list(self.make_perturber(regenerate_metadata=True).perturb(frames=iter(self.make_frames(n=3))))

        assert not np.array_equal(results[0].image, results[1].image)
        assert not np.array_equal(results[1].image, results[2].image)

    def test_frame_metadata_overrides_generated(self) -> None:
        """Values supplied on the frame win over the generated ones."""
        supplied = list(self.make_perturber().perturb(frames=iter(self.make_frames(n=1, metadata=FIXED_METADATA))))
        generated = list(self.make_perturber().perturb(frames=iter(self.make_frames(n=1))))

        assert not np.array_equal(supplied[0].image, generated[0].image)

    def test_supplied_metadata_is_seed_independent(self) -> None:
        """With every section supplied, no random values are used, so the seed cannot matter."""
        frames_a = self.make_frames(n=1, metadata=FIXED_METADATA)
        frames_b = self.make_frames(n=1, metadata=FIXED_METADATA)
        result_a = list(self.make_perturber(seed=1).perturb(frames=iter(frames_a)))[0]
        result_b = list(self.make_perturber(seed=2).perturb(frames=iter(frames_b)))[0]

        assert np.array_equal(result_a.image, result_b.image)

    def test_missing_metadata_renders_without_generation(self) -> None:
        """With generation off, absent metadata still renders — as placeholders."""
        inst = self.make_perturber(generate_missing_metadata=False)
        result = list(inst.perturb(frames=iter(self.make_frames(n=1))))[0]
        generated = list(self.make_perturber().perturb(frames=iter(self.make_frames(n=1))))[0]

        assert result.image.any(), "expected placeholder burn-ins to still be drawn"
        assert not np.array_equal(result.image, generated.image)

    def test_partial_metadata_is_filled_in(self) -> None:
        """Sections absent from a frame are filled from the generated set, not skipped."""
        supplied: MISBST1909Metadata = {"classification_releasability": "UNCLASSIFIED"}
        # perturb() draws its generated set before the first frame, so a perturber built
        # with the same seed produces exactly the values the partial render will fill with.
        generated: MISBST1909Metadata = self.make_perturber()._generate_metadata()
        merged: MISBST1909Metadata = generated | supplied

        filled_in = list(
            self.make_perturber().perturb(frames=iter(self.make_frames(n=1, metadata=supplied))),
        )[0].image
        explicit = list(
            self.make_perturber(generate_missing_metadata=False).perturb(
                frames=iter(self.make_frames(n=1, metadata=merged)),
            ),
        )[0].image
        all_generated = list(
            self.make_perturber().perturb(frames=iter(self.make_frames(n=1))),
        )[0].image

        assert np.array_equal(filled_in, explicit), "absent sections should render from the generated set"
        # Guards the above from passing trivially had the supplied banner matched the generated one.
        assert not np.array_equal(filled_in, all_generated), "the supplied banner should override the generated one"
        # The banner is anchored to the top edge, so every other burn-in must be untouched.
        below_banner = slice(int(0.15 * SMALL_HEIGHT), None)
        assert np.array_equal(filled_in[below_banner], all_generated[below_banner])

    def test_color_is_respected(self) -> None:
        """Burn-ins are drawn in the configured color."""
        inst = self.make_perturber(color=(128, 64, 32), outline_color=None)
        result = list(inst.perturb(frames=iter(self.make_frames(n=1, metadata=FIXED_METADATA))))[0]

        assert (result.image.reshape(-1, 3) == (128, 64, 32)).all(axis=1).any(), "expected the exact configured colour"
        assert result.image[:, :, 1].max() <= 64

    def test_outline_is_drawn_in_outline_color(self) -> None:
        """An outline color paints pixels the body color never would."""
        body_only = self.make_perturber(color=(255, 255, 255), outline_color=None)
        outlined = self.make_perturber(color=(255, 255, 255), outline_color=(255, 0, 0))

        frames_a = self.make_frames(n=1, metadata=FIXED_METADATA)
        frames_b = self.make_frames(n=1, metadata=FIXED_METADATA)
        plain = list(body_only.perturb(frames=iter(frames_a)))[0].image
        ringed = list(outlined.perturb(frames=iter(frames_b)))[0].image

        # White-on-black leaves the channels equal everywhere; a red outline breaks that.
        assert np.array_equal(plain[:, :, 0], plain[:, :, 1])
        assert not np.array_equal(ringed[:, :, 0], ringed[:, :, 1]), "expected red outline pixels"

    def test_font_path_is_used(self, tmp_path: Path) -> None:
        """A supplied ``font_path`` is loaded from disk instead of the bundled default.

        Pillow ships its default face as bytes rather than a file, so a real file is
        written from those same bytes. Note that this does not render identically to the
        default: ``load_default`` pins ``Layout.BASIC`` while ``truetype`` picks Raqm when
        Pillow has it, and the two lay text out slightly differently.
        """
        default_font = ImageFont.load_default(size=10)
        assert isinstance(default_font, ImageFont.FreeTypeFont)
        # Pillow holds the embedded face in a BytesIO rather than on disk.
        assert isinstance(default_font.path, io.BytesIO)

        font_file = tmp_path / "bundled.otf"
        font_file.write_bytes(default_font.path.getvalue())

        inst = self.make_perturber(font_path=str(font_file))

        assert inst.font_path == str(font_file)
        assert inst.get_config()["font_path"] == str(font_file)
        # The loaded font reports the file it came from, so the path was read, not ignored.
        assert inst._load_font(size=20.0).path == str(font_file)
        assert list(inst.perturb(frames=iter(self.make_frames(n=1, metadata=FIXED_METADATA))))[0].image.any()

    def test_bad_font_path_fails_at_construction(self) -> None:
        """An unreadable font is reported when the perturber is built, not mid-video."""
        with pytest.raises(OSError, match="cannot open resource"):
            self.make_perturber(font_path="/nonexistent/no-such-font.ttf")

    def test_load_font_reuses_one_object_per_size(self) -> None:
        """A size is loaded once and reused, so FreeType's glyph cache survives between burn-ins.

        Identity is the assertion that matters: an equal-but-distinct font would still render
        the same, yet would rasterize every glyph again on each of the seven text burn-ins.
        """
        inst = self.make_perturber()

        assert inst._load_font(size=24.0) is inst._load_font(size=24.0)
        assert inst._load_font(size=24.0) is not inst._load_font(size=25.0)

    def test_load_font_is_shared_between_perturbers(self) -> None:
        """The cache keys on the font, not the perturber, so equally configured instances share it."""
        assert self.make_perturber()._load_font(size=24.0) is self.make_perturber()._load_font(size=24.0)

    def test_load_font_clamps_sub_pixel_sizes(self) -> None:
        """FreeType rejects sub-pixel sizes, which a small enough frame would otherwise produce."""
        inst = self.make_perturber()

        assert inst._load_font(size=0.1) is inst._load_font(size=1.0)

    def test_load_font_cache_is_bounded(self) -> None:
        """The cache has a fixed ceiling, so a stream of differing frame sizes cannot grow it."""
        inst = self.make_perturber()
        for size in range(1, _FONT_CACHE_SIZE * 4):
            inst._load_font(size=float(size))

        info = _misb_st1909_burn_in_perturber._load_font.cache_info()
        assert info.maxsize == _FONT_CACHE_SIZE
        assert info.currsize <= _FONT_CACHE_SIZE

    @pytest.mark.usefixtures("host_timezone")
    def test_timestamps_do_not_depend_on_host_timezone(self) -> None:
        """Naive timestamps render as UTC regardless of where the machine is."""
        inst = self.make_perturber()
        rendered = inst._format_datetime(value=datetime(year=2026, month=7, day=30, hour=12, minute=0, second=0))
        assert rendered == "2026-07-30T12:00:00.0Z"

    def test_aware_timestamps_are_converted_to_utc(self) -> None:
        """A timestamp that carries an offset is converted rather than assumed to be UTC."""
        inst = self.make_perturber()
        aware = datetime(
            year=2026,
            month=7,
            day=30,
            hour=12,
            minute=0,
            second=0,
            tzinfo=timezone(timedelta(hours=5)),
        )

        assert inst._format_datetime(value=aware) == "2026-07-30T07:00:00.0Z"

    def test_text_size_scales_with_setting(self) -> None:
        """Larger ``text_size`` covers more of the frame."""
        small = self.make_perturber(text_size=1.0 / 40.0)
        large = self.make_perturber(text_size=1.0 / 12.0)

        small_image = list(small.perturb(frames=iter(self.make_frames(n=1, metadata=FIXED_METADATA))))[0].image
        large_image = list(large.perturb(frames=iter(self.make_frames(n=1, metadata=FIXED_METADATA))))[0].image

        assert large_image.any(axis=2).sum() > small_image.any(axis=2).sum()

    def test_reticle_follows_metadata_position(self) -> None:
        """``reticle_position`` moves the reticle within the frame."""
        left: MISBST1909Metadata = {**FIXED_METADATA, "reticle_position": (0.25, 0.5)}
        right: MISBST1909Metadata = {**FIXED_METADATA, "reticle_position": (0.75, 0.5)}

        inst = self.make_perturber(generate_missing_metadata=False)
        left_image = list(inst.perturb(frames=iter(self.make_frames(n=1, metadata=left))))[0].image
        right_image = list(inst.perturb(frames=iter(self.make_frames(n=1, metadata=right))))[0].image

        assert not np.array_equal(left_image, right_image)

    def test_regression(
        self,
        psnr_mp4_snapshot: SnapshotAssertion,
        ssim_mp4_snapshot: SnapshotAssertion,
    ) -> None:
        """Regression testing results to detect functional changes."""
        frames = islice(read_video(INPUT_DRONE_VIDEO_FILE_PATH), 5)  # noqa: FKA100 - islice does not accept keyword args
        inst = self.make_perturber()
        # Materialize so both snapshot assertions can re-iterate over the same frames.
        results = list(perturber_assertions(perturb=inst, frames=frames))

        psnr_mp4_snapshot.assert_match(iter(results))
        ssim_mp4_snapshot.assert_match(iter(results))

    def test_regression_with_fixed_metadata(
        self,
        psnr_mp4_snapshot: SnapshotAssertion,
        ssim_mp4_snapshot: SnapshotAssertion,
    ) -> None:
        """Regression testing results to detect functional changes."""
        frames = islice(read_video(INPUT_DRONE_VIDEO_FILE_PATH), 5)  # noqa: FKA100 - islice does not accept keyword args
        inst = self.make_perturber()
        frames_with_metadata = [
            VideoFrame(
                image=frame.image,
                timestamp=frame.timestamp,
                boxes=frame.boxes,
                additional_params={"misb_st1909": FIXED_METADATA},
            )
            for frame in frames
        ]
        # Materialize so both snapshot assertions can re-iterate over the same frames.
        results = list(perturber_assertions(perturb=inst, frames=iter(frames_with_metadata)))

        psnr_mp4_snapshot.assert_match(iter(results))
        ssim_mp4_snapshot.assert_match(iter(results))
