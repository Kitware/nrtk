"""Defines MISBST1909BurnInPerturber for rendering MISB ST 1909 metadata burn-ins onto video.

MISB ST 1909 specifies where a motion imagery system paints human-readable sensor and
platform metadata over the image itself, so the information needed to construct the overlay
travels nondestructively alongside the image in the KLV stream. This module reproduces that
overlay as a video perturbation, letting a model be tested against imagery that is partially
occluded by the burn-in a real full-motion-video feed would carry.

Classes:
    MISBST1909BurnInPerturber: A video perturber that draws the ST 1909 burn-in set
        (sensor, classification, platform, laser, timestamps, target, reticle, next-zoom
        box, and north arrow) over each frame using Pillow.
    MISBST1909Metadata: Typed description of the metadata the burn-ins are rendered from,
        along with the per-section TypedDicts it nests.

Example usage:
    >>> perturber = MISBST1909BurnInPerturber(  # doctest: +SKIP
    ...     seed=42,
    ... )
    >>> for frame in perturber(frames=video_frames):  # doctest: +SKIP
    ...     # process perturbed frame
    ...     pass

    Metadata is read from each frame's ``additional_params["misb_st1909"]``. Any section
    absent there is filled in with plausible random values when
    ``generate_missing_metadata`` is set, so the perturber is usable against footage that
    carries no metadata of its own.
"""

import math
from collections.abc import Callable, Generator, Iterator, Sequence
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Literal, TypedDict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.impls.perturb_video._base.numpy_random_perturb_video import NumpyRandomPerturbVideo
from nrtk.interfaces import VideoFrame
from nrtk.interfaces._perturb_video import _perturb_guard

# Pillow's ImageDraw has no antialiasing, so the vector burn-ins are rasterized into a
# coverage mask at this multiple and box-downsampled. Measured: 4x differs from 8x by
# 0.68/255 on a stroked circle, so it is converged; 2x is visibly under-sampled.
_SUPERSAMPLE = 4

_FONT_CACHE_SIZE = 8


@lru_cache(maxsize=_FONT_CACHE_SIZE)
def _load_font(*, font_path: str | None, size: float) -> ImageFont.FreeTypeFont:
    """Load a font face at a pixel size.

    Cached at module scope rather than per instance so that perturbers stay picklable —
    Pillow's font objects are not — which keeps them usable across process pools.

    Args:
        font_path: Path to a font file, or None for Pillow's bundled Aileron Regular.
        size: Em size in pixels.

    Returns:
        The font at that size.

    Raises:
        RuntimeError: If Pillow lacks FreeType, without which the anchored, stroked text
            this perturber draws is unsupported.
    """
    if font_path is not None:
        return ImageFont.truetype(font=font_path, size=size)

    # Returns the bitmap ImageFont only when Pillow lacks FreeType.
    font = ImageFont.load_default(size=size)
    if not isinstance(font, ImageFont.FreeTypeFont):
        msg = "Rendering burn-in text requires a Pillow build with FreeType support."
        raise RuntimeError(msg)
    return font


class MISBST1909MainSensor(TypedDict, total=False):
    """Imaging sensor identity and pointing, rendered in the top-left burn-in.

    Attributes:
        name: Sensor name as displayed on the first line of the burn-in.
        azimuth_degrees: Sensor azimuth relative to the platform, in degrees.
        elevation_degrees: Sensor elevation relative to the platform, in degrees.
    """

    name: str
    azimuth_degrees: float
    elevation_degrees: float


class MISBST1909Platform(TypedDict, total=False):
    """Platform identity and position, rendered in the top-right burn-in.

    Attributes:
        name: Platform name as displayed on the first line of the burn-in.
        latitude_degrees: Platform latitude in degrees, positive north.
        longitude_degrees: Platform longitude in degrees, positive east.
        altitude_meters: Platform height above the WGS-84 ellipsoid (HAE), in meters.
    """

    name: str
    latitude_degrees: float
    longitude_degrees: float
    altitude_meters: float


class MISBST1909LaserSensor(TypedDict, total=False):
    """Laser designator state, rendered in the bottom-left burn-in.

    Attributes:
        name: Laser sensor name as displayed on the first line of the burn-in.
        status: Whether the laser is firing; shown as "Laser ON" or "Laser OFF".
        prf_code: Pulse repetition frequency code, conventionally four digits (0-8).
    """

    name: str
    status: bool
    prf_code: str


class MISBST1909DateTime(TypedDict, total=False):
    """The pair of timestamps rendered in the bottom-center burn-in.

    Both are displayed as ISO 8601 in UTC. They are reported separately because
    metadata is not necessarily sampled at the instant the frame was captured.

    Attributes:
        frame_date_time: Capture time of the frame itself, shown on the "FT" line.
        metadata_date_time: Time the metadata was sampled, shown on the "MT" line.
    """

    frame_date_time: datetime
    metadata_date_time: datetime


class MISBST1909Target(TypedDict, total=False):
    """Target and field-of-view measurements, rendered in the bottom-right burn-in.

    Attributes:
        slant_range_meters: Line-of-sight distance from sensor to target, in meters.
        target_width_meters: Width of the target scene, in meters.
        hfov_degrees: Horizontal field of view, in degrees.
        vfov_degrees: Vertical field of view, in degrees.
        target_latitude_degrees: Frame-center latitude in degrees, positive north.
        target_longitude_degrees: Frame-center longitude in degrees, positive east.
        target_altitude_meters: Frame-center elevation, in meters.
    """

    slant_range_meters: float
    target_width_meters: float
    hfov_degrees: float
    vfov_degrees: float
    target_latitude_degrees: float
    target_longitude_degrees: float
    target_altitude_meters: float


class MISBST1909Metadata(TypedDict, total=False):
    """Full set of metadata the burn-ins are rendered from.

    Supplied per frame via ``VideoFrame.additional_params["misb_st1909"]``. Every key is
    optional in practice: each burn-in falls back to "N/A" for values it cannot find, and
    ``MISBST1909BurnInPerturber`` can fill absent sections with random values.

    Attributes:
        main_sensor: Sensor identity and pointing (top-left).
        classification_releasability: Classification and releasability banner (top-center).
        platform: Platform identity and position (top-right).
        north_angle_degrees: Bearing of true north in the image, in degrees clockwise;
            drives the rotation of the north arrow on the left edge.
        next_zoom_box: Region the sensor will zoom to next, in normalized frame
            coordinates; drawn as four corner brackets.
        laser_sensor: Laser designator state (bottom-left).
        date_time: Frame and metadata timestamps (bottom-center).
        target: Target and field-of-view measurements (bottom-right).
        reticle_position: Reticle center in normalized frame coordinates, where
            ``(0.5, 0.5)`` is the middle of the frame.
    """

    main_sensor: MISBST1909MainSensor
    classification_releasability: str
    platform: MISBST1909Platform
    north_angle_degrees: float
    next_zoom_box: AxisAlignedBoundingBox
    laser_sensor: MISBST1909LaserSensor
    date_time: MISBST1909DateTime
    target: MISBST1909Target
    reticle_position: tuple[float, float]


class MISBST1909BurnInPerturber(NumpyRandomPerturbVideo):
    """Renders the MISB ST 1909 metadata overlay onto each frame of a video.

    Nine burn-ins are drawn per frame: six blocks of text anchored to the frame's corners
    and edges, plus three vector elements — a center reticle, corner brackets marking the
    next zoom region, and a north arrow on the left edge. Text is rasterized by FreeType
    through Pillow; the vector elements are supersampled into a coverage mask before being
    blended, since Pillow's own drawing primitives are not antialiased.

    Metadata comes from each frame's ``additional_params["misb_st1909"]``. Sections that
    are missing are filled in with random values when ``generate_missing_metadata`` is
    set, which makes the perturber usable against footage carrying no metadata at all.
    Random values are drawn from the seeded generator inherited from
    ``NumpyRandomPerturbVideo``, so a fixed ``seed`` yields a reproducible overlay.

    Every size is expressed as a fraction of frame height rather than in pixels, so the
    overlay scales with resolution. Colors are RGB triples matching the frame's channel
    order.

    Note:
        Input frames may be grayscale or RGB; grayscale is widened to three channels, and
        output is always RGB. Frames are never modified in place.

    Attributes:
        generate_missing_metadata:
            Whether to substitute random values for absent metadata sections.
        regenerate_metadata:
            Whether the random substitutes are redrawn per frame rather than held fixed.
        font_path:
            Path to the font file used for all burn-in text, or None for Pillow's
            bundled default.
        text_size:
            Text height as a fraction of frame height.
        color:
            RGB color of the text and shapes.
        outline_color:
            RGB color drawn behind the text and shapes for contrast, or None for no
            outline.
        zoom_box_corner_length:
            Length of each zoom-box corner bracket, as a fraction of the box's own size.
        reticle_outer_size:
            Outer extent of the reticle arms, as a fraction of frame height.
        reticle_inner_size:
            Size of the gap at the reticle's center, as a fraction of frame height.
        line_thickness:
            Stroke width in pixels for the reticle, zoom box, and north arrow.
        north_circle_radius:
            Radius of the north arrow's circle, as a fraction of frame height.
    """

    def __init__(
        self,
        *,
        generate_missing_metadata: bool = True,
        regenerate_metadata: bool = True,
        font_path: str | None = None,
        text_size: float = 1.0 / 32.0,
        color: tuple[int, int, int] = (255, 255, 255),
        outline_color: tuple[int, int, int] | None = (0, 0, 0),
        zoom_box_corner_length: float = 0.1,
        reticle_outer_size: float = 0.1,
        reticle_inner_size: float = 0.03,
        line_thickness: int = 2,
        north_circle_radius: float = 0.05,
        seed: int | None = None,
    ) -> None:
        """Initialize the MISBST1909BurnInPerturber.

        Args:
            generate_missing_metadata:
                When True (the default), any metadata section absent from a frame is
                replaced with plausible random values, so the full overlay is drawn even
                for footage that carries no metadata. When False, missing values render
                as "N/A" instead.
            regenerate_metadata:
                When True (the default), the random substitutes are redrawn for every frame, making the
                overlay flicker with new values as the video plays. When False, one set is drawn per
                ``perturb()`` call and reused for all of that call's frames, which reads as a static
                overlay. Only has an effect when ``generate_missing_metadata`` is set.
            font_path:
                Path to a TrueType or OpenType font file for all burn-in text. When None
                (the default), Pillow's bundled Aileron Regular is used, which renders
                identically on every machine. Pass a path to pick a different face; bold
                and italic are selected by pointing at the corresponding file.
            text_size:
                Text height as a fraction of frame height, so the overlay scales with
                resolution. Default 1/32.
            color:
                RGB color of the text and shapes. Default white.
            outline_color:
                RGB color drawn behind the text and shapes so they stay legible over
                bright imagery, or None to draw no outline. Default black.
            zoom_box_corner_length:
                Length of each corner bracket of the next-zoom box, as a fraction of that
                box's own width and height. At 1.0 the brackets meet and the box is drawn
                closed.
            reticle_outer_size:
                Outer extent of the reticle's arms, as a fraction of frame height.
            reticle_inner_size:
                Size of the gap left open at the reticle's center, as a fraction of frame
                height. Should be smaller than ``reticle_outer_size``.
            line_thickness:
                Stroke width in pixels for the reticle, zoom box, and north arrow. The
                outline, when enabled, has a width of one pixel.
            north_circle_radius:
                Radius of the north arrow's circle, as a fraction of frame height. The
                arrow itself is sized from this radius, so it scales along with the circle.
            seed:
                Random seed for the generated metadata. None for non-deterministic.
        """
        super().__init__(seed=seed)

        self.generate_missing_metadata = generate_missing_metadata
        self.regenerate_metadata = regenerate_metadata
        self.font_path = font_path
        self.text_size = text_size
        self.color = color
        self.outline_color = outline_color
        self.zoom_box_corner_length = zoom_box_corner_length
        self.reticle_outer_size = reticle_outer_size
        self.reticle_inner_size = reticle_inner_size
        self.line_thickness = line_thickness
        self.north_circle_radius = north_circle_radius

        self._fill = (*self.color, 255)
        self._stroke = (*outline_color, 255) if outline_color is not None else None
        # Two pixels, not one: a single-pixel band is thinner than a pixel once antialiasing
        # splits it across a boundary, so no pixel reaches the full outline color and the
        # outline reads as a faint smudge. See ``_stroke_passes`` for the same reasoning.
        self._stroke_width = 2 if outline_color is not None else 0

        # Fail fast on an unreadable path rather than partway through the first frame.
        self._load_font(size=10.0)

    def _load_font(self, *, size: float) -> ImageFont.FreeTypeFont:
        """Return the configured font at a pixel size, from the bounded shared cache.

        Args:
            size: Requested em size in pixels. FreeType rejects sub-pixel sizes, which a
                small enough frame would otherwise produce, so the value is clamped to 1.

        Returns:
            The font at that size.
        """
        return _load_font(font_path=self.font_path, size=max(size, 1.0))

    def _copy_image(self, image: np.ndarray) -> np.ndarray:
        """Converts an image's shape to [H, W, 3] and returns a copy.

        Args:
            image: numpy array from a VideoFrame

        Returns:
                A copy of the image with shape [H, W, 3]
        """
        if image.ndim == 2:
            image = np.repeat(image[:, :, np.newaxis], repeats=3, axis=2)
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        else:
            image = image.copy()

        return image

    @override
    @_perturb_guard
    def perturb(
        self,
        *,
        frames: Iterator[VideoFrame],
        **kwargs: Any,
    ) -> Generator[VideoFrame, None, None]:
        """Draw the ST 1909 burn-ins over each frame of a video.

        Metadata is read per frame from ``additional_params["misb_st1909"]``. When
        ``generate_missing_metadata`` is set, that mapping is layered over a randomly
        generated one, so frame-supplied values win and absent sections are filled in.

        Args:
            frames:
                Iterator over input video frames.
                Frames may be grayscale or RGB.
            kwargs:
                Additional perturbation parameters (not used).

        Yields:
            Perturbed VideoFrame objects, always RGB, with metadata passed through
            unchanged.
        """
        _generated_metadata = self._generate_metadata()
        for frame in frames:
            original_dtype = frame.image.dtype
            # Copy either way: the burn-ins are drawn in place and inputs must be left alone.
            image = self._copy_image(frame.image.astype(np.uint8))

            metadata: MISBST1909Metadata = frame.additional_params.get("misb_st1909", {})
            if self.generate_missing_metadata:
                metadata = _generated_metadata | metadata
                if self.regenerate_metadata:
                    _generated_metadata = self._generate_metadata()

            self._render_burn_ins(image=image, metadata=metadata)

            yield VideoFrame(
                image=image.astype(original_dtype),
                timestamp=frame.timestamp,
                boxes=deepcopy(frame.boxes),
                additional_params=deepcopy(frame.additional_params),
            )

    def _generate_metadata(self) -> MISBST1909Metadata:
        """Generate a complete set of plausible random metadata.

        Values are sampled from the seeded generator over each field's physically
        meaningful range, so the resulting overlay looks like a real feed's without
        describing any real scene.

        Returns:
            A fully populated metadata mapping.
        """
        return {
            "main_sensor": {
                "name": self._random_string(max_chars=24),
                "azimuth_degrees": self._rng.uniform(low=0.0, high=360.0),
                "elevation_degrees": self._rng.uniform(low=-180.0, high=180.0),
            },
            "classification_releasability": self._random_string(max_chars=24),
            "platform": {
                "name": self._random_string(max_chars=24),
                "latitude_degrees": self._rng.uniform(low=-90.0, high=90.0),
                "longitude_degrees": self._rng.uniform(low=-180.0, high=180.0),
                "altitude_meters": self._rng.uniform(low=-900.0, high=10000.0),
            },
            "north_angle_degrees": self._rng.uniform(low=0.0, high=360.0),
            "next_zoom_box": self._random_zoom_box(),
            "laser_sensor": {
                "name": self._random_string(max_chars=24),
                "status": bool(self._rng.choice([True, False])),
                # prf_code only uses 0-8
                "prf_code": "".join([self._rng.choice(list(map(str, range(9)))) for _ in range(4)]),
            },
            "date_time": {
                "frame_date_time": self._random_datetime(),
                "metadata_date_time": self._random_datetime(),
            },
            "target": {
                "slant_range_meters": self._rng.uniform(low=0.0, high=10000.0),
                "hfov_degrees": self._rng.uniform(low=0.0, high=90.0),
                "vfov_degrees": self._rng.uniform(low=0.0, high=90.0),
                "target_width_meters": self._rng.uniform(low=1.0, high=100.0) ** 2,
                "target_latitude_degrees": self._rng.uniform(low=-90.0, high=90.0),
                "target_longitude_degrees": self._rng.uniform(low=-180.0, high=180.0),
                "target_altitude_meters": self._rng.uniform(low=-900.0, high=10000.0),
            },
            "reticle_position": (self._rng.uniform(low=0.3, high=0.7), self._rng.uniform(low=0.3, high=0.7)),
        }

    def _random_string(self, *, max_chars: int) -> str:
        """Build a random string of printable ASCII.

        Args:
            max_chars: Upper bound on the length; the actual length is drawn from
                ``[1, max_chars]``.

        Returns:
            A string of randomly chosen printable ASCII characters.
        """
        length = self._rng.integers(low=1, high=max_chars + 1)
        return "".join([chr(self._rng.integers(low=32, high=127)) for _ in range(length)])

    def _random_datetime(self) -> datetime:
        """Generate a random naive timestamp between 1990 and 2050.

        Days are capped at 28 so every month is valid without special-casing.

        Returns:
            A timestamp with microsecond precision.
        """
        return datetime(
            year=self._rng.integers(low=1990, high=2050, endpoint=True),
            month=self._rng.integers(low=1, high=12, endpoint=True),
            day=self._rng.integers(low=1, high=28, endpoint=True),
            hour=self._rng.integers(low=0, high=24),
            minute=self._rng.integers(low=0, high=60),
            second=self._rng.integers(low=0, high=60),
            microsecond=self._rng.integers(low=0, high=1000000),
        )

    def _random_zoom_box(self) -> AxisAlignedBoundingBox:
        """Generate a random next-zoom box in normalized frame coordinates.

        The corners are sampled so the box always straddles the frame center and stays
        well inside the edges, keeping its brackets clear of the corner text burn-ins.

        Returns:
            A bounding box with vertices in [0, 1].
        """
        lower = self._rng.uniform(low=0.2, high=0.5, size=(2,))
        upper = self._rng.uniform(low=0.5, high=0.8, size=(2,))
        return AxisAlignedBoundingBox(min_vertex=lower, max_vertex=upper)

    # Ignore C901 since function is too complex and cannot be reduced cleanly
    def _render_text(  # noqa: C901
        self,
        *,
        text: str,
        position: tuple[int, int],
        image: np.ndarray,
        x_anchor: Literal["left", "center", "right"] = "left",
        y_anchor: Literal["top", "center", "bottom"] = "top",
        text_align: Literal["left", "center", "right"] = "left",
    ) -> None:
        """Draw a block of text, optionally outlined, anchored to a point.

        Newlines start a new line, stacked downward at the font's natural line spacing.
        Anchoring positions the block as a whole, while alignment positions the lines
        within it; the two are independent, so right-anchored text can still be
        left-aligned.

        Args:
            text: The text to draw. May contain newlines.
            position: Point in pixels the block is anchored to.
            image: Frame being drawn on, used to scale the text to its height.
            x_anchor: Which edge of the block ``position`` fixes horizontally.
            y_anchor: Which edge of the block ``position`` fixes vertically. "top" fixes
                the ascender of the first line and "bottom" the baseline of the last.
            text_align: How lines shorter than the widest one are distributed within the
                block.
        """
        font = self._load_font(size=image.shape[0] * self.text_size)
        ascent, descent = font.getmetrics()
        spacing = ascent + descent

        lines = text.split("\n")
        line_widths = [font.getlength(line) for line in lines]
        total_width = max(line_widths)
        total_height = ascent + spacing * (len(lines) - 1)

        if x_anchor == "center":
            x_offset = -total_width / 2
        elif x_anchor == "right":
            x_offset = -total_width
        else:
            x_offset = 0.0

        if y_anchor == "center":
            y_offset = -total_height / 2
        elif y_anchor == "bottom":
            y_offset = -total_height
        else:
            y_offset = 0.0

        # Each line is placed by its baseline, matching the "ls" anchor used when drawing below.
        baselines = []
        for index, line_width in enumerate(line_widths):
            if text_align == "center":
                align_offset = (total_width - line_width) / 2
            elif text_align == "right":
                align_offset = total_width - line_width
            else:
                align_offset = 0.0
            baselines.append(
                (
                    position[0] + x_offset + align_offset,
                    position[1] + y_offset + ascent + spacing * index,
                ),
            )

        # Render into a tile just big enough for the glyph ink rather than wrapping the whole
        # frame, so each burn-in costs in proportion to its text and not to the resolution.
        ink = [font.getbbox(line, stroke_width=self._stroke_width, anchor="ls") for line in lines]
        corners = [(x + b[0], y + b[1], x + b[2], y + b[3]) for (x, y), b in zip(baselines, ink, strict=True)]
        left = math.floor(min(c[0] for c in corners))
        top = math.floor(min(c[1] for c in corners))
        right = math.ceil(max(c[2] for c in corners))
        bottom = math.ceil(max(c[3] for c in corners))
        if right <= left or bottom <= top:
            return

        tile = Image.new(mode="RGBA", size=(right - left, bottom - top), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(im=tile)
        for (x, y), line in zip(baselines, lines, strict=True):
            draw.text(
                xy=(x - left, y - top),
                text=line,
                font=font,
                fill=self._fill,
                anchor="ls",
                stroke_width=self._stroke_width,
                stroke_fill=self._stroke,
            )

        # Pillow leaves the tile's alpha straight (not premultiplied), so this is a plain
        # source-over blend of the glyph coverage onto the frame.
        tile_array = np.asarray(tile)
        self._composite(
            image=image,
            coverage=tile_array[:, :, 3],
            color=tile_array[:, :, :3],
            origin=(left, top),
        )

    def _composite(
        self,
        *,
        image: np.ndarray,
        coverage: np.ndarray,
        color: np.ndarray | tuple[int, int, int],
        origin: tuple[int, int],
    ) -> None:
        """Blend a color into the frame through a coverage map, clipped to the frame.

        Args:
            image: RGB frame to draw into, modified in place.
            coverage: 2-D uint8 alpha for the region, where 255 is fully opaque.
            color: Either a solid RGB triple or a per-pixel RGB array matching ``coverage``.
            origin: Top-left corner of the region in frame pixels. May be negative.
        """
        left, top = origin
        x0 = max(left, 0)
        y0 = max(top, 0)
        x1 = min(left + coverage.shape[1], image.shape[1])
        y1 = min(top + coverage.shape[0], image.shape[0])
        if x0 >= x1 or y0 >= y1:
            return

        rows = slice(y0 - top, y1 - top)
        cols = slice(x0 - left, x1 - left)
        alpha = coverage[rows, cols, None].astype(np.float32) / 255.0
        source = np.asarray(color, dtype=np.float32)
        if source.ndim == 3:
            source = source[rows, cols]

        region = image[y0:y1, x0:x1, :3]
        region[...] = np.round(source * alpha + region * (1.0 - alpha)).astype(np.uint8)

    def _draw_masked(
        self,
        *,
        image: np.ndarray,
        bounds: tuple[float, float, float, float],
        draw_shapes: Callable[..., None],
        color: tuple[int, int, int],
    ) -> None:
        """Rasterize shapes supersampled, then blend a solid color through their coverage.

        ``ImageDraw`` does not antialias, so the shapes are drawn enlarged into a
        single-channel mask and box-downsampled; averaging whole pixels of that mask is
        what produces the smooth edges. Working in a mask rather than an RGBA tile also
        avoids the dark fringing that downsampling a transparent-black tile would cause.

        Args:
            image: RGB frame to draw into, modified in place.
            bounds: Region to rasterize as ``(left, top, right, bottom)`` in frame pixels.
            draw_shapes: Callback issuing the draw calls, taking keyword-only ``draw``,
                ``scale``, ``left`` and ``top``. It should subtract the offsets from frame
                coordinates and then multiply by the scale.
            color: RGB color to blend through the resulting coverage.
        """
        left = math.floor(bounds[0])
        top = math.floor(bounds[1])
        width = math.ceil(bounds[2]) - left
        height = math.ceil(bounds[3]) - top
        if width <= 0 or height <= 0:
            return

        mask = Image.new(mode="L", size=(width * _SUPERSAMPLE, height * _SUPERSAMPLE), color=0)
        draw_shapes(draw=ImageDraw.Draw(mask), scale=_SUPERSAMPLE, left=left, top=top)
        coverage = np.asarray(mask.resize(size=(width, height), resample=Image.Resampling.BOX))

        self._composite(image=image, coverage=coverage, color=color, origin=(left, top))

    def _format_degrees(self, *, value: float) -> str:
        """Format an angle for display, to four decimal places.

        Args:
            value: Angle in degrees. NaN and infinity render as "N/A".

        Returns:
            The angle with a degree sign, or "N/A".
        """
        return f"{value:.4f}°" if math.isfinite(value) else "N/A"

    def _format_meters(self, *, value: float) -> str:
        """Format a distance for display, rounded to whole meters.

        Args:
            value: Distance in meters. NaN and infinity render as "N/A".

        Returns:
            The distance with a trailing "m", or "N/A".
        """
        return f"{round(value)}m" if math.isfinite(value) else "N/A"

    def _format_datetime(self, *, value: datetime | None) -> str:
        """Format a timestamp as ISO 8601 in UTC, to tenths of a second.

        Naive timestamps are taken to be UTC already. Reading them as local time instead —
        which is what ``astimezone`` does by default — would make the rendered overlay
        depend on the host's timezone, and ST 1909 timestamps are UTC by definition.

        Args:
            value: The timestamp, or None to render "N/A".

        Returns:
            The timestamp as ``YYYY-MM-DDThh:mm:ss.sZ``, or "N/A".
        """
        if value is None:
            return "N/A"
        utc = timezone(timedelta())
        aware = value.replace(tzinfo=utc) if value.tzinfo is None else value
        base = aware.astimezone(utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
        return f"{base[:-5]}Z"

    def _render_main_sensor_burn_in(
        self,
        *,
        image: np.ndarray,
        main_sensor: MISBST1909MainSensor,
    ) -> None:
        """Draw the sensor name and relative pointing angles in the top-left corner.

        Args:
            image: Frame being drawn on, used to place the text.
            main_sensor: Sensor identity and pointing. Missing values render as "N/A".
        """
        name_string = main_sensor.get("name", "N/A")
        azimuth_string = self._format_degrees(value=main_sensor.get("azimuth_degrees", math.nan))
        elevation_string = self._format_degrees(value=main_sensor.get("elevation_degrees", math.nan))

        self._render_text(
            text=f"{name_string}\nREL AZ {azimuth_string}\nREL EL {elevation_string}",
            position=(int(0.02 * image.shape[1]), int(0.037 * image.shape[0])),
            image=image,
            x_anchor="left",
            y_anchor="top",
            text_align="left",
        )

    def _render_classification_burn_in(
        self,
        *,
        image: np.ndarray,
        classification_releasability: str,
    ) -> None:
        """Draw the classification and releasability banner along the top edge.

        Args:
            image: Frame being drawn on, used to place the text.
            classification_releasability: The banner text, centered on the frame.
        """
        self._render_text(
            text=classification_releasability,
            position=(int(0.5 * image.shape[1]), int(0.037 * image.shape[0])),
            image=image,
            x_anchor="center",
            y_anchor="top",
            text_align="center",
        )

    def _render_platform_burn_in(self, *, image: np.ndarray, platform: MISBST1909Platform) -> None:
        """Draw the platform name and geodetic position in the top-right corner.

        Args:
            image: Frame being drawn on, used to place the text.
            platform: Platform identity and position. Missing values render as "N/A".
        """
        name_string = platform.get("name", "N/A")
        latitude_string = self._format_degrees(value=platform.get("latitude_degrees", math.nan))
        longitude_string = self._format_degrees(value=platform.get("longitude_degrees", math.nan))
        altitude_string = self._format_meters(value=platform.get("altitude_meters", math.nan))

        self._render_text(
            text=f"{name_string}\n{latitude_string} LAT\n{longitude_string} LON\n{altitude_string} HAE ALT",
            position=(int(0.98 * image.shape[1]), int(0.037 * image.shape[0])),
            image=image,
            x_anchor="right",
            y_anchor="top",
            text_align="right",
        )

    def _render_laser_sensor_burn_in(
        self,
        *,
        image: np.ndarray,
        laser_sensor: MISBST1909LaserSensor,
    ) -> None:
        """Draw the laser designator's name, state, and PRF code in the bottom-left corner.

        Args:
            image: Frame being drawn on, used to place the text.
            laser_sensor: Laser designator state. An absent status renders as "Laser N/A".
        """
        name_string = laser_sensor.get("name", "Laser")
        status_string = {True: "Laser ON", False: "Laser OFF", None: "Laser N/A"}[laser_sensor.get("status", None)]
        prf_code_string = laser_sensor.get("prf_code", "N/A")

        self._render_text(
            text=f"{name_string}\n{status_string}\nLaser PRF Code {prf_code_string}",
            position=(int(0.02 * image.shape[1]), int(0.963 * image.shape[0])),
            image=image,
            x_anchor="left",
            y_anchor="bottom",
            text_align="left",
        )

    def _render_date_time_burn_in(
        self,
        *,
        image: np.ndarray,
        date_time: MISBST1909DateTime,
    ) -> None:
        """Draw the frame and metadata timestamps along the bottom edge.

        Args:
            image: Frame being drawn on, used to place the text.
            date_time: The two timestamps, labelled "FT" and "MT" respectively.
        """
        frame_string = self._format_datetime(value=date_time.get("frame_date_time", None))
        metadata_string = self._format_datetime(value=date_time.get("metadata_date_time", None))

        self._render_text(
            text=f"FT {frame_string}\nMT {metadata_string}",
            position=(int(0.5 * image.shape[1]), int(0.963 * image.shape[0])),
            image=image,
            x_anchor="center",
            y_anchor="bottom",
            text_align="center",
        )

    def _render_target_burn_in(self, *, image: np.ndarray, target: MISBST1909Target) -> None:
        """Draw the target range, size, field of view, and frame-center position.

        Rendered in the bottom-right corner, one measurement per line.

        Args:
            image: Frame being drawn on, used to place the text.
            target: Target and field-of-view measurements. Missing values render as "N/A".
        """
        text = (
            f"{self._format_meters(value=target.get('slant_range_meters', math.nan))} SR\n"
            f"{self._format_meters(value=target.get('target_width_meters', math.nan))} TW\n"
            f"{self._format_degrees(value=target.get('hfov_degrees', math.nan))} HFOV\n"
            f"{self._format_degrees(value=target.get('vfov_degrees', math.nan))} VFOV\n"
            f"{self._format_degrees(value=target.get('target_latitude_degrees', math.nan))} FC LAT\n"
            f"{self._format_degrees(value=target.get('target_longitude_degrees', math.nan))} FC LON\n"
            f"{self._format_meters(value=target.get('target_altitude_meters', math.nan))} FC EL"
        )
        self._render_text(
            text=text,
            position=(int(0.98 * image.shape[1]), int(0.963 * image.shape[0])),
            image=image,
            x_anchor="right",
            y_anchor="bottom",
            text_align="right",
        )

    def _draw_lines(
        self,
        *,
        image: np.ndarray,
        groups: Sequence[Sequence[tuple[float, float, float, float]]],
    ) -> None:
        """Stroke groups of line segments, outlining them first when an outline is enabled.

        Args:
            image: RGB frame to draw into, modified in place.
            groups: Segments as ``(x0, x1, y0, y1)`` in pixels — note that both x values
                precede both y values — bundled into groups. Segments that touch belong in
                the same group; each group is rasterized in its own tile, sized to fit it.
        """
        # Both passes are drawn in full before the next, so the wider outline of one segment never covers
        # the body of another. Connected segments must share a pass and a mask, or their overlap at the
        # joint would be blended twice and show a seam.
        for color, width in self._stroke_passes():
            extend = (width - self.line_thickness) / 2.0
            for group in groups:
                xs = [v for x0, x1, _, _ in group for v in (x0, x1)]
                ys = [v for _, _, y0, y1 in group for v in (y0, y1)]
                pad = width / 2.0 + extend + 1.0

                def draw_shapes(
                    *,
                    draw: ImageDraw.ImageDraw,
                    scale: int,
                    left: float,
                    top: float,
                    group: Sequence[tuple[float, float, float, float]] = group,
                    width: float = width,
                    extend: float = extend,
                ) -> None:
                    for x0, x1, y0, y1 in group:
                        length = math.hypot(x1 - x0, y1 - y0)
                        step_x = (x1 - x0) / length * extend if length else 0.0
                        step_y = (y1 - y0) / length * extend if length else 0.0
                        draw.line(
                            [
                                ((x0 - step_x - left) * scale, (y0 - step_y - top) * scale),
                                ((x1 + step_x - left) * scale, (y1 + step_y - top) * scale),
                            ],
                            fill=255,
                            width=round(width * scale),
                        )

                self._draw_masked(
                    image=image,
                    bounds=(min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad),
                    draw_shapes=draw_shapes,
                    color=color,
                )

    def _stroke_passes(self) -> list[tuple[tuple[int, int, int], int]]:
        """Return the ``(color, width)`` passes for the vector burn-ins, outline first.

        Both strokes are centred on the path, so the outline shows through as a band of
        ``(outline_width - body_width) / 2`` on each side. That is widened to two pixels
        rather than one because a one-pixel band is thinner than a pixel wherever
        antialiasing splits it across a boundary, leaving no pixel at the full outline
        color; only paths that happen to land grid-aligned looked solid.

        Returns:
            One entry when no outline is configured, otherwise the wider outline pass
            followed by the body pass.
        """
        passes = []
        if self.outline_color is not None:
            passes.append((self.outline_color, self.line_thickness + 4))
        passes.append((self.color, self.line_thickness))
        return passes

    def _render_reticle_burn_in(
        self,
        *,
        image: np.ndarray,
        reticle_position: tuple[float, float],
    ) -> None:
        """Draw the aiming reticle as four arms around an open center.

        Args:
            image: Frame being drawn on, used to scale and place the reticle.
            reticle_position: Reticle center in normalized frame coordinates.
        """
        center_x = image.shape[1] * reticle_position[0]
        center_y = image.shape[0] * reticle_position[1]

        inner_radius = image.shape[0] * self.reticle_inner_size / 2.0
        outer_radius = image.shape[0] * self.reticle_outer_size / 2.0

        lines = [
            (
                round(center_x + inner_radius),
                round(center_x + outer_radius),
                round(center_y),
                round(center_y),
            ),
            (
                round(center_x - inner_radius),
                round(center_x - outer_radius),
                round(center_y),
                round(center_y),
            ),
            (
                round(center_x),
                round(center_x),
                round(center_y + inner_radius),
                round(center_y + outer_radius),
            ),
            (
                round(center_x),
                round(center_x),
                round(center_y - inner_radius),
                round(center_y - outer_radius),
            ),
        ]

        # The four arms never touch, so one group keeps this to a single small tile.
        self._draw_lines(image=image, groups=[lines])

    def _render_zoom_box_burn_in(self, *, image: np.ndarray, box: AxisAlignedBoundingBox) -> None:
        """Draw the next-zoom region as four corner brackets rather than a closed box.

        Args:
            image: Frame being drawn on, used to scale and place the brackets.
            box: The region, in normalized frame coordinates.
        """
        x = image.shape[1] * (box.min_vertex[0] + box.max_vertex[0]) / 2.0
        y = image.shape[0] * (box.min_vertex[1] + box.max_vertex[1]) / 2.0

        w = image.shape[1] * (box.max_vertex[0] - box.min_vertex[0]) / 2.0
        h = image.shape[0] * (box.max_vertex[1] - box.min_vertex[1]) / 2.0

        p = 1.0 - self.zoom_box_corner_length

        lines = [
            (
                round(x + w),
                round(x + w * p),
                round(y + h),
                round(y + h),
            ),
            (
                round(x + w),
                round(x + w),
                round(y + h),
                round(y + h * p),
            ),
            (
                round(x + w),
                round(x + w),
                round(y - h),
                round(y - h * p),
            ),
            (
                round(x + w),
                round(x + w * p),
                round(y - h),
                round(y - h),
            ),
            (
                round(x - w),
                round(x - w * p),
                round(y + h),
                round(y + h),
            ),
            (
                round(x - w),
                round(x - w),
                round(y + h),
                round(y + h * p),
            ),
            (
                round(x - w),
                round(x - w * p),
                round(y - h),
                round(y - h),
            ),
            (
                round(x - w),
                round(x - w),
                round(y - h),
                round(y - h * p),
            ),
        ]

        # The eight segments form four L-shaped corners; each pair meets, so each pair is a group.
        self._draw_lines(image=image, groups=[lines[0:2], lines[2:4], lines[4:6], lines[6:8]])

    def _north_triangle(self, *, north_angle: float, center_x: float, center_y: float, r: float) -> np.ndarray:
        """Build the north arrow's triangle, rotated to point at true north.

        The triangle straddles the circle's rim, reaching from 0.75r to 1.25r so it reads
        as an arrowhead sitting on the circle rather than floating beside it.

        Args:
            north_angle: Bearing of true north, in degrees clockwise.
            center_x: Circle center x in frame pixels.
            center_y: Circle center y in frame pixels.
            r: Circle radius in pixels.

        Returns:
            A ``(3, 2)`` array of vertices in frame pixels.
        """
        triangle_angle = north_angle * math.pi / 180.0
        offset = self.line_thickness / 2.0
        triangle_r0 = r * 0.75 + offset
        triangle_r1 = r * 1.25 + offset
        triangle_arc = 0.4
        triangle = np.array(
            [
                (triangle_r1, 0),
                (triangle_r0, +math.tan(triangle_arc) * (triangle_r0 - triangle_r1)),
                (triangle_r0, -math.tan(triangle_arc) * (triangle_r0 - triangle_r1)),
            ],
        )
        rotation = np.array(
            [
                (math.cos(triangle_angle), -math.sin(triangle_angle)),
                (math.sin(triangle_angle), math.cos(triangle_angle)),
            ],
        )
        return np.matmul(rotation, triangle.transpose()).transpose() + (center_x, center_y)

    def _draw_north_shapes(
        self,
        *,
        image: np.ndarray,
        center_x: float,
        center_y: float,
        r: float,
        triangle: np.ndarray | None,
    ) -> None:
        """Draw the north indicator's circle and arrow through supersampled masks.

        The ring is finished before the arrow starts. Sharing passes between them would let
        the ring's body pass paint over the arrow's outline wherever the two overlap, which
        broke the arrow's border along the rim it sits on.

        Args:
            image: RGB frame to draw into, modified in place.
            center_x: Circle center x in frame pixels.
            center_y: Circle center y in frame pixels.
            r: Circle radius in pixels.
            triangle: Arrow vertices, or None to draw the circle alone.
        """
        reach = r * 1.25 + self.line_thickness / 2.0 + 4.0
        bounds = (center_x - reach, center_y - reach, center_x + reach, center_y + reach)

        self._draw_ring(image=image, bounds=bounds, center_x=center_x, center_y=center_y, r=r)
        if triangle is not None:
            self._draw_arrow(image=image, bounds=bounds, triangle=triangle)

    def _draw_ring(
        self,
        *,
        image: np.ndarray,
        bounds: tuple[float, float, float, float],
        center_x: float,
        center_y: float,
        r: float,
    ) -> None:
        """Stroke the north indicator's circle, outline pass first.

        Args:
            image: RGB frame to draw into, modified in place.
            bounds: Region to rasterize, in frame pixels.
            center_x: Circle center x in frame pixels.
            center_y: Circle center y in frame pixels.
            r: Circle radius in pixels.
        """
        for color, width in self._stroke_passes():

            def draw_shapes(
                *,
                draw: ImageDraw.ImageDraw,
                scale: int,
                left: float,
                top: float,
                width: float = width,
            ) -> None:
                # ImageDraw grows `width` inward from the box, while a stroke is centred on
                # the radius, so the box is inflated by half the width to compensate.
                half = width * scale / 2.0
                draw.ellipse(
                    [
                        (center_x - r - left) * scale - half,
                        (center_y - r - top) * scale - half,
                        (center_x + r - left) * scale + half,
                        (center_y + r - top) * scale + half,
                    ],
                    outline=255,
                    width=round(width * scale),
                )

            self._draw_masked(image=image, bounds=bounds, draw_shapes=draw_shapes, color=color)

    def _draw_arrow(
        self,
        *,
        image: np.ndarray,
        bounds: tuple[float, float, float, float],
        triangle: np.ndarray,
    ) -> None:
        """Draw the north indicator's arrow, outlined on the first pass and filled on the last.

        Args:
            image: RGB frame to draw into, modified in place.
            bounds: Region to rasterize, in frame pixels.
            triangle: Arrow vertices in frame pixels.
        """
        passes = self._stroke_passes()
        for index, (color, _) in enumerate(passes):
            outline_pass = len(passes) > 1 and index == 0

            def draw_shapes(
                *,
                draw: ImageDraw.ImageDraw,
                scale: int,
                left: float,
                top: float,
                outline_pass: bool = outline_pass,
            ) -> None:
                points = [(float(px - left) * scale, float(py - top) * scale) for px, py in triangle]
                if outline_pass:
                    # ImageDraw.polygon grows `width` inward, so the fill in the body pass would
                    # bury the outline; a centred line leaves half its width proud instead. The
                    # path is wrapped past both ends because ImageDraw draws no joint at the
                    # first or last vertex, which left the arrow's apex with a bare butt cap.
                    draw.line(
                        [points[2], *points, points[0]],
                        fill=255,
                        width=round(4 * scale),
                        joint="curve",
                    )
                else:
                    draw.polygon(points, fill=255)

            self._draw_masked(image=image, bounds=bounds, draw_shapes=draw_shapes, color=color)

    def _render_north_arrow_burn_in(self, *, image: np.ndarray, north_angle: float | None) -> None:
        """Draw the north indicator on the left edge: a circle labelled "N" with an arrow.

        The arrow is a triangle riding the circle's rim, rotated to point at true north.
        It is drawn in two stages so it reads correctly against the circle: its outline
        goes down before the circle's body, and its fill after.

        Args:
            image: Frame being drawn on, used to scale and place the indicator.
            north_angle: Bearing of true north in degrees clockwise, or None to draw the
                circle and label with no arrow.
        """
        r = image.shape[0] * self.north_circle_radius
        center_x = image.shape[1] * 0.02 + r
        center_y = image.shape[0] * 0.5

        triangle = (
            None
            if north_angle is None
            else self._north_triangle(
                north_angle=north_angle,
                center_x=center_x,
                center_y=center_y,
                r=r,
            )
        )
        self._draw_north_shapes(image=image, center_x=center_x, center_y=center_y, r=r, triangle=triangle)

        self._render_text(
            text="N",
            position=(round(center_x), round(center_y)),
            image=image,
            x_anchor="center",
            y_anchor="center",
            text_align="center",
        )

    def _render_burn_ins(self, *, image: np.ndarray, metadata: MISBST1909Metadata) -> None:
        """Draw all nine burn-ins over a frame, in place.

        Each burn-in rasterizes into a tile sized to its own content and blends that into
        the frame, so cost tracks the overlay rather than the resolution. Text is drawn
        before the vector elements, which therefore sit on top where they overlap.

        Args:
            image: RGB frame to draw into, modified in place.
            metadata: Values to render. Absent sections fall back to defaults.
        """
        self._render_main_sensor_burn_in(image=image, main_sensor=metadata.get("main_sensor", {}))
        self._render_classification_burn_in(
            image=image,
            classification_releasability=metadata.get("classification_releasability", "N/A"),
        )
        self._render_platform_burn_in(image=image, platform=metadata.get("platform", {}))
        self._render_laser_sensor_burn_in(image=image, laser_sensor=metadata.get("laser_sensor", {}))
        self._render_date_time_burn_in(image=image, date_time=metadata.get("date_time", {}))
        self._render_target_burn_in(image=image, target=metadata.get("target", {}))
        self._render_reticle_burn_in(
            image=image,
            reticle_position=metadata.get("reticle_position", (0.5, 0.5)),
        )
        self._render_zoom_box_burn_in(
            image=image,
            box=metadata.get("next_zoom_box", AxisAlignedBoundingBox(min_vertex=(0.25, 0.25), max_vertex=(0.75, 0.75))),
        )
        self._render_north_arrow_burn_in(
            image=image,
            north_angle=metadata.get("north_angle_degrees", None),
        )

    @override
    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration.

        Returns:
            Configuration dictionary with all constructor parameters.
        """
        cfg = super().get_config()
        cfg["generate_missing_metadata"] = self.generate_missing_metadata
        cfg["regenerate_metadata"] = self.regenerate_metadata
        cfg["font_path"] = self.font_path
        cfg["text_size"] = self.text_size
        cfg["color"] = self.color
        cfg["outline_color"] = self.outline_color
        cfg["zoom_box_corner_length"] = self.zoom_box_corner_length
        cfg["reticle_outer_size"] = self.reticle_outer_size
        cfg["reticle_inner_size"] = self.reticle_inner_size
        cfg["line_thickness"] = self.line_thickness
        cfg["north_circle_radius"] = self.north_circle_radius

        return cfg

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Retrieves the default configuration for MISBST1909BurnInPerturber instances."""
        cfg = super().get_default_config()
        cfg["color"] = list(cfg["color"])
        cfg["outline_color"] = list(cfg["outline_color"])
        return cfg
