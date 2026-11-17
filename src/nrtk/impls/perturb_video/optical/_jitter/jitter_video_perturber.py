"""Defines JitterVideoPerturber for physics-based line-of-sight jitter simulation on video.

Classes:
    JitterVideoPerturber: A video perturber that models platform line-of-sight jitter
        source-side as an angular disturbance spectrum (tonal lines on a colored 1/f**beta
        floor) and renders it per frame as inter-frame displacement (with optional roll,
        wide-FOV keystone, and rolling-shutter row displacement) plus intra-frame directional
        smear, splitting the two by the exposure integration time.

Dependencies:
    - numpy for array operations.
    - scipy for FFT convolution and coordinate-map warping.
    - nrtk.interfaces.PerturbVideo as the base interface for perturbation.

Example usage:
    >>> perturber = JitterVideoPerturber.from_platform("helicopter", seed=42)  # doctest: +SKIP
    >>> for frame in perturber(frames=video_frames):  # doctest: +SKIP
    ...     # process perturbed frame
    ...     pass
"""

from __future__ import annotations

__all__ = ["JitterVideoPerturber"]

import warnings
from collections.abc import Generator, Hashable, Iterable, Iterator, Sequence
from copy import deepcopy
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.impls.perturb_video._base.numpy_random_perturb_video import NumpyRandomPerturbVideo
from nrtk.impls.perturb_video.optical._jitter import _presets, _render
from nrtk.impls.perturb_video.optical._jitter.los_motion import LOSMotionModel
from nrtk.interfaces import VideoFrame
from nrtk.interfaces._perturb_video import _perturb_guard

_AUTO_FS_CAP_HZ = 20_000.0  # PSF path samples its own fine grid, so the buffer fs can be bounded
_PSF_MIN_SAMPLES = 64  # smear-path samples per exposure window (coarse paths collapse the kernel to spikes)


def _is_normalized_float(*, image: np.ndarray[Any, Any]) -> bool:
    """Whether ``image`` is a float array with every value in ``[0, 1]``."""
    return bool(np.issubdtype(image.dtype, np.floating) and image.size and image.min() >= 0.0 and image.max() <= 1.0)


class JitterVideoPerturber(NumpyRandomPerturbVideo):
    """Perturb a video with platform line-of-sight jitter (tip/tilt + optional roll).

    The jitter source is an angular spectrum in physical units (radians, Hz); per frame the
    trajectory is windowed over the exposure ``integration_time``, splitting each frequency's
    energy into displacement (weight ``sinc^2(f*T_int)``) and directional smear
    (``1 - sinc^2``). Displacement + roll (+ per-row rolling-shutter displacement + wide-FOV
    keystone) render as one coordinate-map warp; smear renders as a convolution with the
    within-exposure path kernel (Boracchi & Foi 2012). Bounding boxes move through the same
    map as the image. Preset tonal frequencies are cited; default ``residual_rms``
    amplitudes are representative engineering defaults, not measured platform data.

    Note:
        ``frame.timestamp`` is the exposure START; the window is ``[t, t + integration_time]``.

        Composition: zero pyBSM jitter/drift (``s_x``/``s_y``) when composing with
        ``PybsmPerturber``/``JitterPerturber`` (double-counts LOS motion); recommended
        order is turbulence -> jitter. Real FMV inputs already contain their platform's
        residual jitter, so perturbing them measures *marginal* robustness to *added* jitter.

        Each output frame's ``additional_params["jitter"]`` reports the achieved
        per-frame degradation (dx/dy px, roll, smear extent) — plot scores against this
        measured pixel motion rather than the assumed angular amplitude. A downstream
        jitter perturber replaces the record rather than accumulating it.

        Integer frames are rounded and clipped to their dtype's range. Float frames that
        come in normalized to ``[0, 1]`` are clipped back to ``[0, 1]``, since cubic resampling
        overshoots at hard edges.

    Attributes:
        tones: ``(frequency_Hz, relative_power_weight)`` source lines (``None`` -> a single
            20 Hz line so the bare constructor visibly perturbs).
        broadband_beta: Colored-floor exponent (1/f**beta).
        broadband_weight: Fraction of variance in the floor (0..1).
        residual_rms: Per-axis tip/tilt LOS true RMS in radians (buffer-grid contract).
        severity: Multiplier on ``residual_rms`` (the T&E sweep knob).
        roll_ratio: Roll RMS as a fraction of tip/tilt (image rotation; negligible at narrow FOV).
        pixel_pitch: Detector pitch (m); with ``focal_length`` sets ``ifov``.
        focal_length: Effective focal length (m).
        integration_time: Exposure T_int (s); drives the smear/displacement split.
        rolling_shutter_readout_s: CMOS full-frame row-readout time (s); None = global shutter.
        sub_pixel: Cubic (True, order=3) vs nearest (False, order=0) warp interpolation.
        color_fill: Edge fill for revealed pixels (None = 0), as in TurbulenceVideoPerturber.
        internal_fs: Trajectory buffer sample rate (Hz); None -> auto, capped at 20 kHz.
        hf_cutoff_hz: Optional extra floor low-pass (reshapes, does not remove, floor energy).
        max_duration_s: Floor buffer length (s) and the RMS-normalization span; timestamps
            beyond raise ValueError.
        seed: RNG seed; None = non-deterministic per call.

    Examples:
        >>> p = JitterVideoPerturber(tones=[(20.0, 1.0)], residual_rms=1.5e-5, seed=0)
        >>> ifov = p.pixel_pitch / p.focal_length
        >>> bool(ifov > 0)
        True
    """

    def __init__(
        self,
        *,
        tones: Sequence[Sequence[float]] | None = None,
        broadband_beta: float = 2.0,
        broadband_weight: float = 0.1,
        residual_rms: float = 15e-6,
        severity: float = 1.0,
        roll_ratio: float = 1.0,
        pixel_pitch: float = 5e-6,
        focal_length: float = 0.5,
        integration_time: float = 1e-3,
        rolling_shutter_readout_s: float | None = None,
        sub_pixel: bool = True,
        color_fill: int | Sequence[int] | None = None,
        internal_fs: float | None = None,
        hf_cutoff_hz: float | None = None,
        max_duration_s: float = 30.0,
        seed: int | None = None,
    ) -> None:
        """Initialize the JitterVideoPerturber.

        Args:
            tones:
                ``(frequency_Hz, relative_power_weight)`` source lines. ``None`` (the
                default) selects a single 20 Hz line so the bare constructor visibly
                perturbs; the default is ``None`` rather than a tuple so the signature
                stays JSON-serializable for configuration round-trips.
            broadband_beta:
                Colored-floor exponent (power spectral density proportional to 1/f**beta).
            broadband_weight:
                Fraction of total variance carried by the floor, in [0, 1].
            residual_rms:
                Per-axis tip/tilt LOS true RMS in radians, enforced exactly over the
                trajectory buffer grid.
            severity:
                Multiplier on ``residual_rms``; the natural T&E sweep parameter.
            roll_ratio:
                Roll-about-boresight RMS as a fraction of tip/tilt. Rendered as image
                rotation; numerically negligible at narrow FOV.
            pixel_pitch:
                Detector pixel pitch (m). With ``focal_length`` sets
                ``ifov = pixel_pitch / focal_length`` (rad/pixel), the angle-to-pixel
                mapping.
            focal_length:
                Effective focal length (m). See ``pixel_pitch``.
            integration_time:
                Exposure time T_int (s); drives the smear/displacement split.
            rolling_shutter_readout_s:
                CMOS full-frame row-readout time (s); each image row samples the
                trajectory at its own exposure start. ``None`` = global shutter.
            sub_pixel:
                If True (default), warp with cubic interpolation (order=3); if False,
                nearest-neighbor (order=0).
            color_fill:
                Fill value for edge pixels revealed by displacement (``None`` = 0), with
                the same semantics as ``TurbulenceVideoPerturber``.
            internal_fs:
                Trajectory buffer sample rate (Hz). ``None`` derives
                ``max(4 x max tone, 10 / T_int, 1000)`` capped at 20 kHz (the smear-PSF
                path samples its own finer grid).
            hf_cutoff_hz:
                Optional extra low-pass on the floor, in Hz; must be below ``internal_fs / 2``
                (the resolved rate, see ``internal_fs``). The floor is re-standardized after
                filtering, so the cutoff reshapes rather than removes floor energy.
            max_duration_s:
                Length (s) of the seeded floor buffer; the commanded RMS is enforced over
                ``[0, max_duration_s]``. For an exact delivered amplitude set this to the
                clip length plus ``integration_time`` (plus ``rolling_shutter_readout_s``
                if set); larger values also work but loosen the delivered clip RMS on
                floor-heavy configurations, since a short clip then samples only part of
                the normalization span. Frame timestamps needing samples beyond the buffer
                raise ValueError rather than silently freezing the floor.
            seed:
                Random seed for reproducibility. None for non-deterministic. The floor
                realization also depends on the buffer geometry — ``max_duration_s`` and
                ``internal_fs`` (auto-derived from ``integration_time`` and the tone set)
                — so pin those explicitly when sweeping them if the floor must not
                re-roll between sweep points.

        Raises:
            ValueError: If any parameter is outside its documented range, or is NaN.
        """
        if tones is None:
            tones = ((20.0, 1.0),)  # a visible single-tone default; None keeps the signature JSON-stable
        self.tones: Sequence[tuple[float, float]] = [(float(f), float(w)) for f, w in tones]
        self.broadband_beta = broadband_beta
        self.broadband_weight = broadband_weight
        self.residual_rms = residual_rms
        self.severity = severity
        self.roll_ratio = roll_ratio
        self.pixel_pitch = pixel_pitch
        self.focal_length = focal_length
        self.integration_time = integration_time
        self.rolling_shutter_readout_s = rolling_shutter_readout_s
        self.sub_pixel = sub_pixel
        self.color_fill = color_fill
        self.hf_cutoff_hz = hf_cutoff_hz
        self.max_duration_s = max_duration_s
        self._validate(internal_fs=internal_fs)
        self._warn_degenerate()
        max_tone = max((f for f, _ in self.tones), default=0.0)
        auto_fs = max(4.0 * max_tone, 10.0 / integration_time, 1000.0)
        self.internal_fs: float = float(internal_fs) if internal_fs is not None else min(auto_fs, _AUTO_FS_CAP_HZ)
        if self.hf_cutoff_hz is not None and not self.hf_cutoff_hz < self.internal_fs / 2.0:
            # Checked here rather than in _validate because it needs the resolved sample rate;
            # otherwise scipy raises an opaque error from inside perturb().
            raise ValueError(
                f"hf_cutoff_hz={self.hf_cutoff_hz} must be below half the trajectory sample rate "
                f"(internal_fs / 2 = {self.internal_fs / 2.0} Hz); lower hf_cutoff_hz or raise internal_fs",
            )
        super().__init__(seed=seed)

    def _validate(self, *, internal_fs: float | None) -> None:  # noqa: C901
        """Check constructor parameters against their documented ranges.

        Raises:
            ValueError: If any parameter is out of range, or is NaN.
        """
        if not self.broadband_beta >= 0:
            raise ValueError("broadband_beta must be non-negative")
        if not 0.0 <= self.broadband_weight <= 1.0:
            raise ValueError("broadband_weight must be in [0, 1]")
        if not self.residual_rms >= 0:
            raise ValueError("residual_rms must be non-negative")
        if not self.severity >= 0:
            raise ValueError("severity must be non-negative")
        if not self.roll_ratio >= 0:
            raise ValueError("roll_ratio must be non-negative")
        if not self.pixel_pitch > 0:
            raise ValueError("pixel_pitch must be positive")
        if not self.focal_length > 0:
            raise ValueError("focal_length must be positive")
        if not self.integration_time > 0:
            raise ValueError("integration_time must be positive")
        if self.rolling_shutter_readout_s is not None and not self.rolling_shutter_readout_s > 0:
            raise ValueError("rolling_shutter_readout_s must be positive (or None for global shutter)")
        if internal_fs is not None and not internal_fs > 0:
            raise ValueError("internal_fs must be positive (or None for auto)")
        if self.hf_cutoff_hz is not None and not self.hf_cutoff_hz > 0:
            raise ValueError("hf_cutoff_hz must be positive (or None to disable)")
        if not self.max_duration_s > 0:
            raise ValueError("max_duration_s must be positive")
        if any(not f > 0 for f, _ in self.tones):
            raise ValueError("tone frequencies must be positive")
        if any(not w >= 0 for _, w in self.tones):
            raise ValueError("tone weights must be non-negative")

    def _warn_degenerate(self) -> None:
        """Warn when there is no motion source, or when the RMS normalization makes a parameter inert."""
        if any(w > 0 for _, w in self.tones):
            return
        if self.broadband_weight == 0.0:
            warnings.warn(
                "tones carry no energy and broadband_weight=0: there is no motion source, so frames "
                "pass through unperturbed.",
                UserWarning,
                stacklevel=3,
            )
        elif self.broadband_weight < 1.0:
            warnings.warn(
                "tones carry no energy: the RMS normalization rescales the floor to carry ALL the "
                "residual energy, so broadband_weight has no effect. Set broadband_weight=1.0 "
                "to silence this warning.",
                UserWarning,
                stacklevel=3,
            )

    @override
    def get_config(self) -> dict[str, Any]:
        """Return the current configuration (all constructor parameters).

        Returns:
            Configuration dictionary with all constructor parameters.
        """
        cfg = super().get_config()  # brings "seed" from RandomPerturbVideo
        cfg["tones"] = [list(t) for t in self.tones]
        cfg["broadband_beta"] = self.broadband_beta
        cfg["broadband_weight"] = self.broadband_weight
        cfg["residual_rms"] = self.residual_rms
        cfg["severity"] = self.severity
        cfg["roll_ratio"] = self.roll_ratio
        cfg["pixel_pitch"] = self.pixel_pitch
        cfg["focal_length"] = self.focal_length
        cfg["integration_time"] = self.integration_time
        cfg["rolling_shutter_readout_s"] = self.rolling_shutter_readout_s
        cfg["sub_pixel"] = self.sub_pixel
        cfg["color_fill"] = list(self.color_fill) if isinstance(self.color_fill, Sequence) else self.color_fill
        cfg["internal_fs"] = self.internal_fs
        cfg["hf_cutoff_hz"] = self.hf_cutoff_hz
        cfg["max_duration_s"] = self.max_duration_s
        return cfg

    @classmethod
    def from_platform(
        cls,
        platform: str,
        *,
        severity: float = 1.0,
        mode: str | None = None,
        seed: int | None = None,
        **overrides: Any,
    ) -> JitterVideoPerturber:
        """Build a perturber from a platform preset; ``overrides`` win over preset defaults.

        Preset tonal frequencies are cited; default ``residual_rms`` amplitudes are
        representative engineering defaults, not measured platform data. Call
        ``platform_provenance()`` for sources and the shape-provisional flag; pass
        ``residual_rms=`` to override.

        Args:
            platform: One of :meth:`list_platforms` (``maritime`` and ``handheld`` take
                ``mode=``: fine/coarse and unstabilized/ois respectively).
            severity: Multiplier on the preset residual RMS (the sweep knob).
            mode: Sub-mode for platforms that define one.
            seed: RNG seed.
            overrides: Any constructor parameter, overriding the preset value.

        Returns:
            Configured JitterVideoPerturber.

        Raises:
            ValueError: Unknown platform, unknown mode, a mode passed for a mode-less
                platform, or an override that is out of range for the constructor.

        Examples:
            >>> p = JitterVideoPerturber.from_platform("helicopter", seed=0)
            >>> len(p.tones) > 0
            True
        """
        spec = _presets.resolve_preset(platform=platform, mode=mode)
        params: dict[str, Any] = {
            "tones": list(spec.tones),
            "broadband_beta": spec.broadband_beta,
            "broadband_weight": spec.broadband_weight,
            "residual_rms": spec.residual_rms,
            "roll_ratio": spec.roll_ratio,
            "pixel_pitch": spec.pixel_pitch,
            "focal_length": spec.focal_length,
            "integration_time": spec.integration_time,
            "rolling_shutter_readout_s": spec.rolling_shutter_readout_s,
            "severity": severity,
            "seed": seed,
        }
        params.update(overrides)
        return cls(**params)

    @staticmethod
    def list_platforms() -> list[str]:
        """Return the available preset platform names (mode-bearing platforms listed once).

        Examples:
            >>> "helicopter" in JitterVideoPerturber.list_platforms()
            True
        """
        return _presets.list_platforms()

    @staticmethod
    def platform_provenance(platform: str, *, mode: str | None = None) -> dict[str, Any]:
        """Return the preset's audit record.

        Keys: ``citation``, ``shape_provisional``, ``units``, ``default_residual_rms``
        (radians), ``notes``. Amplitudes are representative engineering defaults across
        all presets, never measured platform data.

        Raises:
            ValueError: Unknown platform, unknown mode, or a mode passed for a
                mode-less platform.

        Examples:
            >>> JitterVideoPerturber.platform_provenance("helicopter")["shape_provisional"]
            False
        """
        spec = _presets.resolve_preset(platform=platform, mode=mode)
        return {
            "citation": spec.citation,
            "shape_provisional": spec.shape_provisional,
            "units": spec.units,
            "default_residual_rms": spec.residual_rms,
            "notes": spec.notes,
        }

    def _build_model(self) -> LOSMotionModel:
        return LOSMotionModel(
            tones=self.tones,
            broadband_beta=self.broadband_beta,
            broadband_weight=self.broadband_weight,
            residual_rms_rad=self.residual_rms * self.severity,
            roll_ratio=self.roll_ratio,
            internal_fs=self.internal_fs,
            hf_cutoff_hz=self.hf_cutoff_hz,
            max_duration_s=self.max_duration_s,
        )

    def _row_displacements(
        self,
        *,
        model: LOSMotionModel,
        ts: float,
        height: int,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], float]:
        """Per-output-row displacement (px) and window-mean roll (rad) for one frame.

        Window means use the midpoint rule with a fixed sample count
        ``n = round(T_int * internal_fs)``: the count cannot wobble with the timestamp, and
        the midpoint rule keeps the exact sinc null at integer ``f * T_int`` (an
        endpoint-inclusive mean leaks ~1/n of a pure-smear tone into displacement). Global
        shutter: one window at ``ts``. Rolling shutter: row ``r`` of ``height`` opens at
        ``ts + (r/height) * readout``. Roll uses the frame-start window (per-row roll is
        second-order).
        """
        t_int = self.integration_time
        n_win = max(2, int(round(t_int * self.internal_fs)))
        offsets = (np.arange(n_win) + 0.5) * (t_int / n_win)
        if self.rolling_shutter_readout_s is None:
            starts = np.array([ts])
        else:
            starts = ts + (np.arange(height) / height) * self.rolling_shutter_readout_s
        t_mat = starts[:, None] + offsets[None, :]
        tx, ty, troll = model.sample(t=t_mat.ravel())
        ifov = self.pixel_pitch / self.focal_length
        dx = tx.reshape(t_mat.shape).mean(axis=1) / ifov
        dy = ty.reshape(t_mat.shape).mean(axis=1) / ifov
        roll = float(troll.reshape(t_mat.shape)[0].mean())
        if self.rolling_shutter_readout_s is None:
            dx = np.full(height, dx[0])
            dy = np.full(height, dy[0])
        return dx, dy, roll

    def _smear_psf(self, *, model: LOSMotionModel, ts: float) -> tuple[np.ndarray[Any, Any] | None, float]:
        """Directional smear kernel and its 99th-percentile extent (px) for one frame.

        The path is sampled on its own fine grid (>= ``_PSF_MIN_SAMPLES`` per window,
        decoupled from ``internal_fs``): at ~10 samples/window a one-cycle 3 px sinusoid
        rasterizes to isolated spikes instead of a streak. Under rolling shutter the same
        kernel is shared by all rows (row kernels differ only in phase along the trajectory;
        ensemble statistics are row-invariant). The returned extent is 0.0 when the excursion
        is sub-pixel and no kernel is applied.
        """
        n = max(_PSF_MIN_SAMPLES, int(round(self.integration_time * self.internal_fs)))
        t_psf = ts + (np.arange(n) + 0.5) * (self.integration_time / n)
        tx, ty, _ = model.sample(t=t_psf)
        ifov = self.pixel_pitch / self.focal_length
        path_px_x = (tx - tx.mean()) / ifov
        path_px_y = (ty - ty.mean()) / ifov
        psf = _render.directional_smear_psf(path_px_x=path_px_x, path_px_y=path_px_y)
        if psf is None:
            return None, 0.0
        return psf, float(np.percentile(np.hypot(path_px_x, path_px_y), 99))

    def _resolve_fill(self, *, image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Resolve color_fill to a per-channel float array (same semantics as TurbulenceVideoPerturber).

        Raises:
            ValueError: If ``color_fill`` does not have one entry per image channel, or
                holds a non-finite value (which would reach revealed edge pixels and cast
                to an undefined integer).
        """
        expected_channels = image.shape[-1] if image.ndim == 3 else 1
        if self.color_fill is None:
            return np.zeros(expected_channels, dtype=np.float64)
        if isinstance(self.color_fill, int):
            return np.full(expected_channels, float(self.color_fill))
        fill = np.asarray(self.color_fill, dtype=np.float64)
        if fill.size != expected_channels:
            raise ValueError(f"color_fill length {fill.size} does not match image channel count {expected_channels}")
        if not np.isfinite(fill).all():
            raise ValueError("color_fill values must be finite")
        return fill

    def _shift_frame_boxes(
        self,
        *,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]],
        dx_rows: np.ndarray[Any, Any],
        dy_rows: np.ndarray[Any, Any],
        roll_rad: float,
        image_shape: tuple[int, ...],
    ) -> list[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]:
        """Move boxes with the image, using the same inverse map as the warp.

        Boxes keep sub-pixel precision and are clipped to the frame. A box that leaves the
        frame is dropped. Boxes are not grown for smear, which blurs an object about its
        displaced position without moving it.
        """
        boxes = list(boxes)
        if not boxes:
            return []
        coords = _render.source_coords(
            dx_rows=dx_rows,
            dy_rows=dy_rows,
            roll_rad=roll_rad,
            ifov=self.pixel_pitch / self.focal_length,
            height=image_shape[0],
            width=image_shape[1],
            border=True,
        )
        corners = [
            (float(bbox.min_vertex[0]), float(bbox.min_vertex[1]), float(bbox.max_vertex[0]), float(bbox.max_vertex[1]))
            for bbox, _ in boxes
        ]
        out: list[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] = []
        for (_, metadata), bounds in zip(boxes, _render.box_bounds(coords=coords, boxes=corners), strict=True):
            if bounds is None:
                continue
            out.append(
                (AxisAlignedBoundingBox(min_vertex=bounds[:2], max_vertex=bounds[2:]), deepcopy(metadata)),
            )
        return out

    @override
    @_perturb_guard
    def perturb(
        self,
        *,
        frames: Iterator[VideoFrame],
        **additional_params: Any,
    ) -> Generator[VideoFrame, None, None]:
        """Yield jittered frames; re-seeds per call so a set seed reproduces output exactly.

        Args:
            frames: Iterator of VideoFrame (pass ``iter(list)``); ``timestamp`` = exposure
                start, must be strictly increasing and within ``[0, max_duration_s]``.
            additional_params: Unused (interface compatibility).

        Yields:
            Perturbed VideoFrame objects; each carries an ``additional_params["jitter"]``
            record with the achieved per-frame degradation.
        """
        self._set_seed()
        model = self._build_model()
        model.reset(rng=self._rng)
        order = 3 if self.sub_pixel else 0
        static = self.residual_rms * self.severity == 0.0
        for frame in frames:
            self._check_frame_time(ts=frame.timestamp)
            if static:
                # Match the perturbing path's validation and telemetry so a severity
                # sweep behaves uniformly at 0: bad color_fill still raises, and the
                # degradation record is present (all zeros) at every sweep point.
                self._resolve_fill(image=frame.image)
                extra = deepcopy(frame.additional_params)
                extra["jitter"] = {
                    "dx_px": 0.0,
                    "dy_px": 0.0,
                    "roll_rad": 0.0,
                    "smear_extent_px": 0.0,
                    "rms_commanded_px": 0.0,
                }
                yield VideoFrame(
                    image=frame.image.copy(),
                    timestamp=frame.timestamp,
                    boxes=deepcopy(list(frame.boxes)),
                    additional_params=extra,
                )
                continue
            yield self._perturb_frame(frame=frame, model=model, order=order)

    def _check_frame_time(self, *, ts: float) -> None:
        """Validate a frame's exposure interval against the trajectory buffer bounds.

        Applied on every frame — including the ``severity=0`` fast path — so a severity
        sweep fails on out-of-range timestamps at every sweep point, not only nonzero ones.

        Raises:
            ValueError: If ``ts`` is negative or NaN, or if the exposure (plus rolling
                shutter readout) ends past ``max_duration_s``, where the floor buffer
                would otherwise silently freeze at its edge value.
        """
        if not ts >= 0.0:
            raise ValueError("frame timestamps must be non-negative (re-zero timestamps)")
        end = ts + self.integration_time + (self.rolling_shutter_readout_s or 0.0)
        if end > self.max_duration_s:
            raise ValueError(
                f"frame at {ts:.3f}s needs trajectory samples past max_duration_s={self.max_duration_s}; "
                f"raise max_duration_s or re-zero frame timestamps",
            )

    def _perturb_frame(self, *, frame: VideoFrame, model: LOSMotionModel, order: int) -> VideoFrame:
        height = frame.image.shape[0]
        dx_rows, dy_rows, roll_rad = self._row_displacements(model=model, ts=frame.timestamp, height=height)
        psf, smear_extent = self._smear_psf(model=model, ts=frame.timestamp)
        # Stay float between the convolution and the warp; the single rint+clip cast at
        # the end avoids quantizing integer frames twice (half an LSB of rounding per
        # pass, which matters on low-contrast imagery).
        image = frame.image.astype(np.float64)
        if psf is not None:
            image = _render.convolve_psf(image=image, psf=psf)
        ifov = self.pixel_pitch / self.focal_length
        image = _render.warp_frame(
            image=image,
            dx_rows=dx_rows,
            dy_rows=dy_rows,
            roll_rad=roll_rad,
            ifov=ifov,
            order=order,
            fill=self._resolve_fill(image=frame.image),
        )
        image = _render.cast_back(out=image, dtype=frame.image.dtype)
        if _is_normalized_float(image=frame.image):
            # the cubic warp overshoots at hard edges
            image = np.clip(image, 0.0, 1.0)
        boxes = self._shift_frame_boxes(
            boxes=frame.boxes,
            dx_rows=dx_rows,
            dy_rows=dy_rows,
            roll_rad=roll_rad,
            image_shape=frame.image.shape,
        )
        extra = deepcopy(frame.additional_params)
        extra["jitter"] = {
            "dx_px": float(dx_rows.mean()),
            "dy_px": float(dy_rows.mean()),
            "roll_rad": roll_rad,
            "smear_extent_px": smear_extent,
            "rms_commanded_px": self.residual_rms * self.severity / ifov,
        }
        return VideoFrame(image=image, timestamp=frame.timestamp, boxes=boxes, additional_params=extra)
