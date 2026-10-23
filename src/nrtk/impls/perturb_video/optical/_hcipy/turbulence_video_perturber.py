"""Defines TurbulenceVideoPerturber for physics-based atmospheric turbulence simulation on video.

Classes:
    TurbulenceVideoPerturber: A video perturber that simulates atmospheric turbulence
        via temporally evolving blur and jitter from a wind-advected HCIPy phase screen.

Dependencies:
    - numpy for array operations.
    - scipy for FFT convolution and interpolation.
    - hcipy for atmospheric phase screen simulation and wavefront propagation.
    - nrtk.interfaces.PerturbVideo as the base interface for perturbation.

Example usage:
    >>> perturber = TurbulenceVideoPerturber(  # doctest: +SKIP
    ...     path_avg_cn2=1.7e-14,
    ...     slant_range=1000.0,
    ...     pixel_pitch=8e-6,
    ...     focal_length=0.04,
    ...     seed=42,
    ... )
    >>> for frame in perturber(frames=video_frames):  # doctest: +SKIP
    ...     # process perturbed frame
    ...     pass
"""

from __future__ import annotations

__all__ = ["TurbulenceVideoPerturber"]

import warnings
from collections.abc import Generator, Hashable, Iterable, Iterator, Sequence
from copy import deepcopy
from typing import Any

import numpy as np
from hcipy import (
    Field,
    FraunhoferPropagator,
    InfiniteAtmosphericLayer,
    Wavefront,
    make_circular_aperture,
    make_focal_grid,
    make_obstructed_circular_aperture,
    make_pupil_grid,
)
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import shift as ndi_shift
from scipy.signal import fftconvolve
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.impls.perturb_video._base.numpy_random_perturb_video import NumpyRandomPerturbVideo
from nrtk.interfaces import VideoFrame
from nrtk.interfaces._perturb_video import _perturb_guard

_MAX_NUM_AIRY = 150
_FRIED_COEFFICIENT = 0.423
_AUTO_GRID_MIN = 32
_AUTO_GRID_MAX = 512
_PITCH_PER_R0_TARGET = 8  # auto-grid samples r0 with pitch = r0/8 (Lane 1992)
_PITCH_PER_R0_WARN = 6  # warn the user when explicit grid_size has pitch > r0/6


def _next_power_of_two(x: int) -> int:
    """Smallest power of two >= max(x, 1)."""
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


class TurbulenceVideoPerturber(NumpyRandomPerturbVideo):
    """Simulates atmospheric turbulence effects on video using HCIPy.

    Produces temporally evolving blur and jitter from a wind-advected
    phase screen. A single turbulent layer models the integrated path
    turbulence; the phase screen decorrelates on a timescale of order
    D / wind_speed for tilt and r0 / wind_speed for higher-order modes.

    The HCIPy state (pupil grid, aperture, atmospheric layer, focal
    grid, and Fraunhofer propagator) is constructed once in
    ``__init__`` and reused across ``perturb()`` calls. Each call
    resets the layer to its t=0 state. When ``seed`` is set, this
    produces bit-identical output across repeated calls. When ``seed``
    is ``None``, each call produces a statistically independent
    non-reproducible sequence.

    Note:
        At extreme turbulence (D/r0 > 60), the focal grid size is
        capped to limit memory use, which may clip the outer wings
        of the seeing-limited PSF.

    Attributes:
        path_avg_cn2:
            Path-averaged refractive index structure parameter
            (m^(-2/3)), treated as spatially uniform along the slant range.
        slant_range:
            Optical path length from observer to target (m).
        D:
            Aperture diameter (m).
        pixel_pitch:
            Detector pixel pitch (m).
        focal_length:
            Optical focal length (m).
        eta:
            Central obscuration ratio [0, 1).
        wavelength:
            Observation wavelength (m).
        wind_speed:
            Wind speed (m/s).
        wind_direction_deg:
            Wind direction in degrees; None = random.
        L0:
            Von Kármán outer scale (m).
        grid_size:
            Pupil plane grid resolution.
        color_fill:
            Edge fill value for jitter shifts.
        sub_pixel:
            If True, apply jitter via cubic-spline sub-pixel shift;
            if False (default), round to nearest integer pixel.
        r0:
            Fried parameter (m). Read-only property computed at init.
        d_over_r0:
            Ratio D/r0. Read-only property.
        ifov:
            Instantaneous field of view = pixel_pitch / focal_length (rad/pixel). Read-only property.
    """

    def __init__(
        self,
        *,
        path_avg_cn2: float = 1.7e-14,
        slant_range: float = 1000.0,
        D: float = 40e-3,  # noqa: N803
        pixel_pitch: float = 8e-6,
        focal_length: float = 0.04,
        eta: float = 0.0,
        wavelength: float = 0.55e-6,
        wind_speed: float = 5.0,
        wind_direction_deg: float | None = None,
        L0: float = 10.0,  # noqa: N803
        grid_size: int | None = None,
        color_fill: int | Sequence[int] | None = None,
        sub_pixel: bool = False,
        seed: int | None = None,
    ) -> None:
        """Initialize the TurbulenceVideoPerturber.

        Args:
            path_avg_cn2:
                Path-averaged refractive index structure parameter Cn²
                (m^(-2/3)), treated as spatially uniform along the slant range.
                The Fried-parameter integral collapses to ``Cn² · slant_range``
                under this approximation, so this argument is the constant
                Cn² used in that integral. Default 1.7e-14 corresponds to a
                weak-turbulence path-averaged value.
            slant_range:
                Total optical path length from observer to target (m).
            D:
                Aperture diameter of the imaging system (m).
            pixel_pitch:
                Detector pixel pitch (m). Together with ``focal_length``
                fixes the camera's instantaneous field of view,
                ``ifov = pixel_pitch / focal_length`` (rad/pixel), which
                sets the angular-to-pixel mapping of the simulated PSF
                and jitter. Ground sample distance is then a derived
                quantity: ``img_gsd = ifov * slant_range``.
            focal_length:
                Optical focal length (m). See ``pixel_pitch``.
            eta:
                Central obscuration ratio (0 for unobstructed aperture).
            wavelength:
                Observation wavelength (m). Default 0.55e-6 (visible band center).
            wind_speed:
                Wind speed (m/s). Controls temporal evolution rate.
                When 0, the atmosphere is frozen (no temporal evolution).
            wind_direction_deg:
                Wind direction in degrees (0=East, 90=North). None for random.
            L0:
                Von Kármán outer scale (m). Default 10.0.
            grid_size:
                Pupil plane grid resolution (pixels per side). When ``None``
                (the default), an appropriate value is auto-derived from
                ``8 * D / r0`` rounded up to the next power of two and
                clamped to ``[32, 512]``. This resolves the screen at
                Lane-Glindemann-Dainty 1992's ``pitch <= r0/8`` criterion
                without paying for excess pixels at weak turbulence. Pass
                an explicit ``int`` to override the auto-sizing; a
                ``UserWarning`` is emitted if the chosen value undersamples r0.
            color_fill:
                Fill value for edges exposed by jitter shifts.
            sub_pixel:
                If True, render atmospheric jitter with sub-pixel precision
                using cubic-spline interpolation (``scipy.ndimage.shift`` at
                ``order=3``). If False (the default), round the jitter to the
                nearest integer pixel. Sub-pixel is ~100-200x slower per frame
                at 720p+ but preserves fractional-pixel tilt that would
                otherwise round to zero, which matters at weak turbulence
                (jitter rms < 1 px) and for T&E fidelity.
            seed:
                Random seed for reproducibility. None for non-deterministic.
        """
        # Store all params BEFORE super().__init__(), which calls _set_seed()
        self.path_avg_cn2 = path_avg_cn2
        self.slant_range = slant_range
        self.D = D
        self.pixel_pitch = pixel_pitch
        self.focal_length = focal_length
        self.eta = eta
        self.wavelength = wavelength
        self.wind_speed = wind_speed
        self.wind_direction_deg = wind_direction_deg
        self.L0 = L0
        self.color_fill = color_fill
        self.sub_pixel = sub_pixel

        # r0 is needed before _validate so the grid_size auto-resolution and
        # sampling warning can both reference it.
        self._r0 = self.compute_r0(path_avg_cn2=path_avg_cn2, wavelength=wavelength, slant_range=slant_range)

        self.grid_size: int = self._auto_grid_size() if grid_size is None else grid_size

        self._validate()
        self._warn_if_undersampled()

        # _ifov is computed after _validate so a non-positive focal_length
        # surfaces as a clear ValueError rather than ZeroDivisionError.
        self._ifov = pixel_pitch / focal_length

        super().__init__(seed=seed)

        # Built once; per-seed reproducibility relies on layer.reset() in perturb().
        self._initialize_atmosphere()

    @property
    def r0(self) -> float:
        """Fried parameter r0 (m), characterizing turbulence strength."""
        return self._r0

    @property
    def d_over_r0(self) -> float:
        """Ratio of aperture diameter to Fried parameter (D/r0)."""
        return self.D / self._r0

    @property
    def ifov(self) -> float:
        """Instantaneous field of view (rad/pixel), = pixel_pitch / focal_length."""
        return self._ifov

    @staticmethod
    def compute_r0(
        *,
        path_avg_cn2: float,
        wavelength: float,
        slant_range: float,
    ) -> float:
        """Compute the Fried parameter r0 from atmospheric conditions.

        The Fried parameter is the aperture diameter over which the
        RMS wavefront phase error is approximately 1 radian. Larger
        r0 means weaker turbulence.

        Args:
            path_avg_cn2:
                Path-averaged Cn² (m^(-2/3)), treated as
                spatially uniform along the slant range.
            wavelength:
                Observation wavelength (m).
            slant_range:
                Total optical path length (m).

        Returns:
            Fried parameter r0 (m).

        Example:
            >>> TurbulenceVideoPerturber.compute_r0(
            ...     path_avg_cn2=1.7e-14, wavelength=0.55e-6, slant_range=1000.0
            ... )  # doctest: +SKIP
        """
        if path_avg_cn2 <= 0:
            raise ValueError("path_avg_cn2 must be positive")
        if wavelength <= 0:
            raise ValueError("wavelength must be positive")
        if slant_range <= 0:
            raise ValueError("slant_range must be positive")
        k = 2.0 * np.pi / wavelength
        return float((_FRIED_COEFFICIENT * k**2 * path_avg_cn2 * slant_range) ** (-3.0 / 5.0))

    @staticmethod
    def compute_path_avg_cn2(
        *,
        r0: float,
        wavelength: float,
        slant_range: float,
    ) -> float:
        """Compute path-averaged Cn² from a known Fried parameter.

        Inverts the r0 equation to recover the path-averaged refractive
        index structure parameter under the spatially-uniform-Cn²
        approximation.

        Args:
            r0: Fried parameter (m).
            wavelength: Observation wavelength (m).
            slant_range: Total optical path length (m).

        Returns:
            Path-averaged Cn² (m^(-2/3)).

        Example:
            >>> cn2 = TurbulenceVideoPerturber.compute_path_avg_cn2(  # doctest: +SKIP
            ...     r0=0.05,
            ...     wavelength=0.55e-6,
            ...     slant_range=1000.0,
            ... )
            >>> perturber = TurbulenceVideoPerturber(path_avg_cn2=cn2, slant_range=1000.0)  # doctest: +SKIP
        """
        if r0 <= 0:
            raise ValueError("r0 must be positive")
        if wavelength <= 0:
            raise ValueError("wavelength must be positive")
        if slant_range <= 0:
            raise ValueError("slant_range must be positive")
        k = 2.0 * np.pi / wavelength
        return float(r0 ** (-5.0 / 3.0) / (_FRIED_COEFFICIENT * k**2 * slant_range))

    def _auto_grid_size(self) -> int:
        """Pick grid_size from r0 via the pitch ≤ r0/8 sampling rule.

        Uses the standard sampling rule for FFT-based Kolmogorov phase
        screens (Schmidt 2010, *Numerical Simulation of Optical Wave
        Propagation*, §9.5; motivated by the phase-screen analysis of
        Lane, Glindemann, Dainty 1992, *Waves in Random Media* 2, 209).
        With ``pitch = D / N``, the requirement becomes ``N ≥ 8 · D / r0``.

        Rounded up to the next power of two for FFT efficiency, clamped to
        ``[32, 512]``: 32 prevents under-resolving the aperture mask itself
        (need ~32 pixels across to resolve a circle), and 512 is the largest
        grid auto-sizing will select without an explicit override. Extreme
        turbulence that wants a bigger grid must pass ``grid_size`` directly.
        """
        target = int(np.ceil(_PITCH_PER_R0_TARGET * self.D / self._r0))
        return max(_AUTO_GRID_MIN, min(_next_power_of_two(target), _AUTO_GRID_MAX))

    def _warn_if_undersampled(self) -> None:
        """Emit a UserWarning if the pupil pitch exceeds r0/6.

        At ``pitch > r0/6`` the FFT phase screen begins to alias
        Kolmogorov structure-function power. Fires whenever the
        configuration is undersampled, whether the small ``grid_size``
        came from an explicit user override or from the auto-sizer
        clamping at its upper bound under extreme turbulence.
        """
        pitch = self.D / self.grid_size
        if pitch > self._r0 / _PITCH_PER_R0_WARN:
            target = int(np.ceil(_PITCH_PER_R0_TARGET * self.D / self._r0))
            recommended = max(_AUTO_GRID_MIN, _next_power_of_two(target))
            warnings.warn(
                f"Pupil pitch {pitch * 1e3:.3f} mm > r0/6 = {self._r0 / _PITCH_PER_R0_WARN * 1e3:.3f} mm; "
                f"Kolmogorov phase screen may be aliased. "
                f"Consider grid_size >= {recommended} (currently {self.grid_size}).",
                UserWarning,
                stacklevel=2,
            )

    def _validate(self) -> None:  # noqa: C901
        """Validate constructor parameters.

        Raises:
            ValueError: If any parameter is out of valid range.
        """
        if self.D <= 0:
            raise ValueError("D must be positive")
        if self.slant_range <= 0:
            raise ValueError("slant_range must be positive")
        if self.path_avg_cn2 <= 0:
            raise ValueError("path_avg_cn2 must be positive")
        if self.pixel_pitch <= 0:
            raise ValueError("pixel_pitch must be positive")
        if self.focal_length <= 0:
            raise ValueError("focal_length must be positive")
        if self.wavelength <= 0:
            raise ValueError("wavelength must be positive")
        if self.wind_speed < 0:
            raise ValueError("wind_speed must be non-negative")
        if self.L0 <= 0:
            raise ValueError("L0 must be positive")
        if self.grid_size < _AUTO_GRID_MIN:
            raise ValueError(f"grid_size must be >= {_AUTO_GRID_MIN}")
        if self.eta < 0 or self.eta >= 1:
            raise ValueError("eta must be in [0, 1)")

    def _initialize_atmosphere(self) -> None:
        """Build static HCIPy state and draw scenario parameters once.

        Called from __init__. The HCIPy seed and (when
        ``wind_direction_deg`` is None) the wind direction are drawn
        here and cached on the instance. Per-call non-determinism comes
        from re-randomizing the phase-screen realization, not the wind.
        """
        # Draw hcipy_seed before the direction so the atmospheric realization
        # is fixed by ``seed`` alone, independent of whether wind_direction_deg
        # is None (which consumes an extra uniform draw).
        hcipy_seed = int(self._rng.integers(low=0, high=2**31))
        if self.wind_direction_deg is None:
            self._direction_rad = float(self._rng.uniform(low=0, high=2 * np.pi))
        else:
            self._direction_rad = float(np.deg2rad(self.wind_direction_deg))
        wind_vx = self.wind_speed * np.cos(self._direction_rad)
        wind_vy = self.wind_speed * np.sin(self._direction_rad)

        pupil_grid = make_pupil_grid(
            dims=self.grid_size,
            diameter=self.D,  # pyright: ignore[reportArgumentType] - HCIPy stub types diameter as int
        )
        if self.eta > 0:
            self._aperture = make_obstructed_circular_aperture(
                pupil_diameter=self.D,
                central_obscuration_ratio=self.eta,
            )(pupil_grid)
        else:
            self._aperture = make_circular_aperture(diameter=self.D)(pupil_grid)
        self._pupil_grid = pupil_grid

        self._layer = InfiniteAtmosphericLayer(
            input_grid=pupil_grid,
            Cn_squared=self.path_avg_cn2 * self.slant_range,
            L0=self.L0,
            velocity=[wind_vx, wind_vy],  # pyright: ignore[reportArgumentType] - HCIPy stub types velocity as int
            height=self.slant_range / 2,  # pyright: ignore[reportArgumentType] - HCIPy stub types height as int
            seed=hcipy_seed,
        )

        # Focal grid spans the seeing disk (~D/r0 Airy radii) with 2.5x margin,
        # capped by _MAX_NUM_AIRY to bound memory.
        requested_airy = int(np.ceil(2.5 * self.D / self._r0))
        num_airy = max(20, min(requested_airy, _MAX_NUM_AIRY))
        if requested_airy > _MAX_NUM_AIRY:
            warnings.warn(
                f"Focal grid num_airy capped at {_MAX_NUM_AIRY} (requested {requested_airy}). "
                f"PSF wings may be clipped at D/r0={self.d_over_r0:.1f}.",
                UserWarning,
                stacklevel=2,
            )

        focal_grid = make_focal_grid(
            q=4,
            num_airy=num_airy,
            pupil_diameter=self.D,
            focal_length=1.0,
            reference_wavelength=self.wavelength,
        )
        self._propagator = FraunhoferPropagator(input_grid=pupil_grid, output_grid=focal_grid)
        self._focal_grid = focal_grid

    def _restore_atmosphere_for_perturb(self) -> None:
        """Reset the cached layer's phase screen to t=0.

        Scenario parameters are fixed at construction. ``seed=None``
        advances to an independent realization each call; ``seed=int``
        restores the construction-time screen for reproducibility.
        """
        self._layer.reset(make_independent_realization=self._seed is None)

    def _extract_tilt(self, wf: Wavefront) -> tuple[float, float]:
        """Extract tip/tilt angles from the PSF centroid in the focal plane.

        Propagates the full wavefront (with tilt) to the focal plane
        and computes the intensity-weighted centroid.

        Args:
            wf: HCIPy Wavefront object.

        Returns:
            Tuple of (tilt_x, tilt_y) angles in radians.
        """
        focal_field = self._propagator(wf)
        psf = np.array(focal_field.power)
        total = psf.sum()
        if total == 0:
            return 0.0, 0.0

        coords = self._focal_grid.coords
        centroid_x = float(np.sum(psf * np.array(coords[0])) / total)
        centroid_y = float(np.sum(psf * np.array(coords[1])) / total)
        return centroid_x, centroid_y

    def _remove_tilt(self, *, wf: Wavefront, tilt_x: float, tilt_y: float) -> Wavefront:
        """Remove tip/tilt from wavefront phase.

        Args:
            wf: HCIPy Wavefront object.
            tilt_x: X-axis tilt angle (radians).
            tilt_y: Y-axis tilt angle (radians).

        Returns:
            Modified Wavefront with tilt removed.
        """
        k = 2.0 * np.pi / self.wavelength
        coords = self._pupil_grid.coords
        tilt_phase = k * (tilt_x * coords[0] + tilt_y * coords[1])

        wf_copy = wf.copy()
        wf_copy.electric_field *= np.exp(-1j * tilt_phase)
        return wf_copy

    def _resample_psf(self, *, psf: Field, plate_scale: float) -> np.ndarray[Any, Any]:
        """Resample PSF from HCIPy focal grid to image plate scale.

        Args:
            psf: HCIPy Field representing the PSF intensity.
            plate_scale: Image plate scale (rad/pixel).

        Returns:
            2D numpy array of the resampled, normalized PSF.
        """
        focal_coords = self._focal_grid.coords
        x_focal = np.array(focal_coords[0])
        y_focal = np.array(focal_coords[1])

        x_unique = np.sort(np.unique(x_focal))
        y_unique = np.sort(np.unique(y_focal))

        psf_flat = np.array(psf)
        if len(y_unique) * len(x_unique) != len(psf_flat):
            raise ValueError(
                "PSF grid structure mismatch: expected product of unique coordinate counts to equal PSF length",
            )
        # HCIPy stores Field values as a flat 1D array; reshape to 2D (y, x)
        # for scipy's RegularGridInterpolator.
        psf_2d = psf_flat.reshape((len(y_unique), len(x_unique)))

        interpolator = RegularGridInterpolator(
            points=(y_unique, x_unique),
            values=psf_2d,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

        half_extent = max(x_unique[-1], y_unique[-1])
        n_pixels = max(1, int(np.ceil(2 * half_extent / plate_scale)) + 1)
        if n_pixels % 2 == 0:
            n_pixels += 1

        # Odd-sized output grid centered on zero keeps the PSF symmetric
        # around the image pixel it's later convolved against.
        half_grid = (n_pixels - 1) // 2
        pixel_coords = (np.arange(n_pixels) - half_grid) * plate_scale
        yy, xx = np.meshgrid(pixel_coords, pixel_coords, indexing="ij")
        points = np.column_stack([yy.ravel(), xx.ravel()])

        resampled = interpolator(points).reshape((n_pixels, n_pixels))

        total = resampled.sum()
        if total > 0:
            resampled /= total

        return resampled

    def _convolve(
        self,
        *,
        image: np.ndarray[Any, Any],
        psf: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Convolve image with PSF using FFT convolution.

        Args:
            image: Input image array (H, W) or (H, W, C).
            psf: 2D PSF kernel.

        Returns:
            Convolved image with same shape and dtype as input.
        """
        img_float = self._convert_image(image=image)

        if img_float.ndim == 2:
            result = fftconvolve(in1=img_float, in2=psf, mode="same")
        else:
            result = np.empty_like(img_float)
            for c in range(img_float.shape[2]):
                result[:, :, c] = fftconvolve(in1=img_float[:, :, c], in2=psf, mode="same")

        return self._restore_image(image=result, original_image=image)

    def _resolve_fill(self, image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Resolve color_fill to match image channel count.

        Args:
            image: Input image to match fill dimensions against.

        Returns:
            Fill value as numpy array.
        """
        expected_channels = image.shape[-1] if image.ndim == 3 else 1
        if self.color_fill is None:
            return np.zeros(expected_channels, dtype=image.dtype)
        if isinstance(self.color_fill, int):
            return np.full(expected_channels, self.color_fill, dtype=image.dtype)
        fill = np.array(self.color_fill, dtype=image.dtype)
        if len(fill) != expected_channels:
            raise ValueError(
                f"color_fill length {len(fill)} does not match image channel count {expected_channels}",
            )
        return fill

    def _apply_shift(
        self,
        *,
        image: np.ndarray[Any, Any],
        shift_x: float,
        shift_y: float,
        original_image: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Apply atmospheric jitter shift to image.

        Dispatches to :meth:`_apply_shift_subpixel` when ``sub_pixel=True``
        (cubic-spline sub-pixel interpolation, physically faithful but slow)
        or :meth:`_apply_shift_integer` otherwise (nearest-integer-pixel
        ``np.roll``, fast but discards fractional tilt).
        """
        if self.sub_pixel:
            return self._apply_shift_subpixel(
                image=image,
                shift_x=shift_x,
                shift_y=shift_y,
                original_image=original_image,
            )
        return self._apply_shift_integer(
            image=image,
            shift_x=shift_x,
            shift_y=shift_y,
            original_image=original_image,
        )

    def _apply_shift_subpixel(
        self,
        *,
        image: np.ndarray[Any, Any],
        shift_x: float,
        shift_y: float,
        original_image: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Sub-pixel shift via cubic-spline interpolation.

        Preserves fractional-pixel displacements that would otherwise round
        to zero under integer rounding. Uses ``scipy.ndimage.shift`` with
        ``order=3`` and ``prefilter=True``; ~100-200x slower per frame than
        the integer path at 720p and above.
        """
        if shift_x == 0.0 and shift_y == 0.0:
            return image.copy()

        fill = self._resolve_fill(original_image)
        img_f64 = self._convert_image(
            image=image,
        )

        if img_f64.ndim == 2:
            shifted = ndi_shift(
                img_f64,
                shift=(shift_y, shift_x),
                order=3,
                mode="constant",
                cval=float(fill[0]),
                prefilter=True,
            )
        else:
            shifted = np.empty_like(img_f64)
            for c in range(img_f64.shape[-1]):
                shifted[:, :, c] = ndi_shift(
                    img_f64[:, :, c],
                    shift=(shift_y, shift_x),
                    order=3,
                    mode="constant",
                    cval=float(fill[c]),
                    prefilter=True,
                )
        return self._restore_image(image=shifted, original_image=image)

    def _apply_shift_integer(  # noqa: C901
        self,
        *,
        image: np.ndarray[Any, Any],
        shift_x: float,
        shift_y: float,
        original_image: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Integer-pixel shift via ``np.roll``.

        Rounds the float shift to the nearest whole pixel. Sub-millisecond
        per frame but discards any shift with ``|shift| < 0.5 px``, so
        weak turbulence (jitter rms ≲ 0.3 px) renders as no visible jitter
        at all. Use ``sub_pixel=True`` on the perturber for faithful
        rendering at low magnitudes.
        """
        translate_x = int(np.round(shift_x))
        translate_y = int(np.round(shift_y))

        # Clamp to image dimensions to prevent np.roll wrap-around artifacts.
        h, w = image.shape[0], image.shape[1]
        translate_x = max(-w, min(translate_x, w))
        translate_y = max(-h, min(translate_y, h))

        if translate_x == 0 and translate_y == 0:
            return image.copy()

        fill = self._resolve_fill(original_image)

        if image.ndim == 3:
            final_image = np.full_like(image, fill.astype(image.dtype), dtype=image.dtype)
        else:
            final_image = np.full_like(image, fill[0], dtype=image.dtype)

        translated = np.roll(image, (translate_y, translate_x), axis=[0, 1])

        if translate_x >= 0 and translate_y >= 0:
            final_image[translate_y:, translate_x:, ...] = translated[translate_y:, translate_x:, ...]
        elif translate_x < 0 and translate_y >= 0:
            final_image[translate_y:, :translate_x, ...] = translated[translate_y:, :translate_x, ...]
        elif translate_x >= 0 and translate_y < 0:
            final_image[:translate_y, translate_x:, ...] = translated[:translate_y, translate_x:, ...]
        else:
            final_image[:translate_y, :translate_x, ...] = translated[:translate_y, :translate_x, ...]

        return final_image

    @staticmethod
    def _shift_boxes(
        *,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]],
        shift_x: float,
        shift_y: float,
        image_shape: tuple[int, ...],
        sub_pixel: bool = False,
    ) -> list[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]:
        """Shift and clip bounding boxes by jitter amount.

        Args:
            boxes: Input bounding boxes.
            shift_x: Horizontal shift in pixels.
            shift_y: Vertical shift in pixels.
            image_shape: Shape of the image (H, W, ...).
            sub_pixel: When True, preserve fractional shift to match the
                sub-pixel image rendering path. When False (default), round
                to the nearest integer pixel to match ``np.roll``.

        Returns:
            Shifted and clipped bounding boxes.
        """
        if sub_pixel:
            translate_x: float = float(shift_x)
            translate_y: float = float(shift_y)
        else:
            translate_x = float(int(np.round(shift_x)))
            translate_y = float(int(np.round(shift_y)))
        h, w = image_shape[0], image_shape[1]

        shifted_boxes: list[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] = []
        for bbox, metadata in boxes:
            new_min_x = max(0, min(bbox.min_vertex[0] + translate_x, w))
            new_min_y = max(0, min(bbox.min_vertex[1] + translate_y, h))
            new_max_x = max(0, min(bbox.max_vertex[0] + translate_x, w))
            new_max_y = max(0, min(bbox.max_vertex[1] + translate_y, h))

            if new_min_x >= new_max_x or new_min_y >= new_max_y:
                continue

            adjusted_box = AxisAlignedBoundingBox(
                min_vertex=(new_min_x, new_min_y),
                max_vertex=(new_max_x, new_max_y),
            )
            shifted_boxes.append((adjusted_box, deepcopy(metadata)))

        return shifted_boxes

    @override
    @_perturb_guard
    def perturb(
        self,
        *,
        frames: Iterator[VideoFrame],
        **kwargs: Any,
    ) -> Generator[VideoFrame, None, None]:
        """Generate turbulence-perturbed video frames.

        The HCIPy pupil grid, aperture, atmospheric layer, focal grid, and
        propagator are built once in ``__init__`` and reused across calls;
        each ``perturb()`` call resets the layer to its t=0 state.
        When ``seed`` is set, identical inputs produce identical outputs.

        The angular-to-pixel mapping is fixed by the ``pixel_pitch`` and
        ``focal_length`` constructor arguments (see ``ifov``); there is
        no per-call plate-scale parameter.

        Args:
            frames:
                Iterator over input video frames. Pass ``iter(my_list)``
                to feed a list. Timestamps drive temporal evolution.
            kwargs:
                Additional perturbation parameters (not used).

        Yields:
            Perturbed VideoFrame objects.
        """
        self._set_seed()
        self._restore_atmosphere_for_perturb()

        plate_scale = self._ifov

        for frame in frames:
            self._layer.evolve_until(frame.timestamp)

            wf = Wavefront(
                electric_field=self._aperture,
                wavelength=self.wavelength,  # pyright: ignore[reportArgumentType] - HCIPy stub types wavelength as int
            )
            wf = self._layer(wf)

            tilt_x, tilt_y = self._extract_tilt(wf)
            wf_no_tilt = self._remove_tilt(wf=wf, tilt_x=tilt_x, tilt_y=tilt_y)
            psf = self._propagator(wf_no_tilt).power
            psf_resampled = self._resample_psf(psf=psf, plate_scale=plate_scale)

            blurred = self._convolve(image=frame.image, psf=psf_resampled)

            pixel_shift_x = tilt_x / plate_scale
            pixel_shift_y = tilt_y / plate_scale

            jittered = self._apply_shift(
                image=blurred,
                shift_x=pixel_shift_x,
                shift_y=pixel_shift_y,
                original_image=frame.image,
            )

            shifted_boxes = self._shift_boxes(
                boxes=frame.boxes,
                shift_x=pixel_shift_x,
                shift_y=pixel_shift_y,
                image_shape=frame.image.shape,
                sub_pixel=self.sub_pixel,
            )

            yield VideoFrame(
                image=jittered,
                timestamp=frame.timestamp,
                boxes=shifted_boxes,
                additional_params=deepcopy(frame.additional_params),
            )

    @override
    def _convert_image(
        self,
        *,
        image: np.ndarray[Any, Any],
        convert_to_rgb: bool = False,
        normalize_floats: bool = False,
    ) -> np.ndarray[Any, Any]:
        """Return a float64 copy without scaling pixel values."""
        working = image.astype(np.float64, copy=True)

        if convert_to_rgb:
            if working.ndim == 2:
                working = np.repeat(working[:, :, np.newaxis], repeats=3, axis=2)
            elif working.shape[2] == 1:
                working = np.repeat(working, repeats=3, axis=2)

        return working

    @override
    def _restore_image(
        self,
        *,
        image: np.ndarray[Any, Any],
        original_image: np.ndarray[Any, Any],
        restore_channels: bool = True,
        normalize_floats: bool = False,
    ) -> np.ndarray[Any, Any]:
        """Restore shape and dtype, clipping integer outputs before casting."""
        height, width = original_image.shape[:2]
        restored = image[:height, :width]

        if restore_channels and restored.ndim == 3:
            if original_image.ndim == 2:
                restored = restored[:, :, 0]
            elif original_image.shape[2] == 1:
                restored = restored[:, :, :1]

        if np.issubdtype(original_image.dtype, np.integer):
            info = np.iinfo(original_image.dtype)
            restored = np.clip(restored, info.min, info.max)

        return restored.astype(original_image.dtype, copy=True)

    @override
    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration.

        Returns:
            Configuration dictionary with all constructor parameters.
        """
        cfg = super().get_config()
        cfg["path_avg_cn2"] = self.path_avg_cn2
        cfg["slant_range"] = self.slant_range
        cfg["D"] = self.D
        cfg["pixel_pitch"] = self.pixel_pitch
        cfg["focal_length"] = self.focal_length
        cfg["eta"] = self.eta
        cfg["wavelength"] = self.wavelength
        cfg["wind_speed"] = self.wind_speed
        cfg["wind_direction_deg"] = self.wind_direction_deg
        cfg["L0"] = self.L0
        cfg["grid_size"] = self.grid_size
        cfg["color_fill"] = list(self.color_fill) if isinstance(self.color_fill, Sequence) else self.color_fill
        cfg["sub_pixel"] = self.sub_pixel
        return cfg
