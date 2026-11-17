"""Defines the source-side line-of-sight (LOS) motion model for the FMV jitter perturber.

Classes:
    LOSMotionModel: Continuous-time 3-axis angular trajectory (tip, tilt, roll) built from
        tonal lines on a colored 1/f**beta broadband floor, normalized to an exact target RMS.

Functions:
    colored_noise: Unit-variance 1/f**beta noise after Timmer & Koenig (1995).
    rotating_element_comb: Tonal comb for rotors/propellers (shaft rate + blade-pass harmonics).

Dependencies:
    - numpy for array operations.
    - scipy for filtering.
"""

from __future__ import annotations

__all__ = ["LOSMotionModel", "rotating_element_comb"]

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import signal


def colored_noise(*, n: int, beta: float, fs: float, rng: np.random.Generator) -> np.ndarray[Any, Any]:
    """Generate unit-variance, zero-mean noise with a 1/f**beta power spectrum.

    Implements the Timmer & Koenig (1995) method: draw a white complex spectrum, shape its
    amplitude by f**(-beta/2), and inverse-FFT. The DC bin is zeroed (zero mean) and the
    Nyquist bin is made real for even ``n``.

    Args:
        n: Number of samples.
        beta: Power-law exponent (1 = pink, 2 = brown/red).
        fs: Sample rate in Hz (sets the frequency grid).
        rng: Seeded numpy Generator (reproducibility).

    Returns:
        1-D array of length ``n``, zero mean, unit standard deviation.

    Examples:
        >>> import numpy as np
        >>> x = colored_noise(n=1024, beta=2.0, fs=100.0, rng=np.random.default_rng(0))
        >>> bool(abs(x.std() - 1.0) < 1e-9)
        True
    """
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    amp = np.zeros_like(freqs)
    amp[1:] = freqs[1:] ** (-beta / 2.0)
    spectrum = (rng.standard_normal(freqs.size) + 1j * rng.standard_normal(freqs.size)) * amp
    spectrum[0] = 0.0
    if n % 2 == 0:
        spectrum[-1] = spectrum[-1].real
    x = np.fft.irfft(spectrum, n=n)
    return (x - x.mean()) / (x.std() + 1e-12)


def rotating_element_comb(
    *,
    fundamental_hz: float,
    n_blades: int,
    n_harmonics: int = 3,
    rolloff_db_oct: float = 3.0,
) -> list[tuple[float, float]]:
    """Build a tonal comb for a rotating element: 1P fundamental + blade-pass harmonics.

    Produces lines at the shaft rate (1P) and at ``k * n_blades * 1P`` for k=1..n_harmonics,
    with weights decaying by ``rolloff_db_oct`` per octave above the blade-pass line. Reusable
    across helicopter, fixed-wing, multirotor, and maritime propeller presets
    (MIL-STD-810H Method 514.8 comb structure).

    Args:
        fundamental_hz: Shaft rotation rate 1P in Hz (= rpm / 60).
        n_blades: Number of blades (blade-pass = n_blades * 1P).
        n_harmonics: Number of blade-pass harmonics to include.
        rolloff_db_oct: Power roll-off above the blade-pass frequency, dB/octave.

    Returns:
        List of ``(frequency_Hz, relative_power_weight)`` tuples.

    Examples:
        >>> comb = rotating_element_comb(fundamental_hz=5.0, n_blades=2, n_harmonics=2)
        >>> [round(f, 2) for f, _ in comb]
        [5.0, 10.0, 20.0]
    """
    bpf = n_blades * fundamental_hz
    tones: list[tuple[float, float]] = [(fundamental_hz, 0.5)]
    for k in range(1, n_harmonics + 1):
        octaves = np.log2(k)  # octaves above blade-pass
        weight = 10.0 ** (-(rolloff_db_oct * octaves) / 10.0)
        tones.append((k * bpf, float(weight)))
    return tones


@dataclass
class _AxisState:
    phases: np.ndarray[Any, Any]
    floor_unit: np.ndarray[Any, Any]
    norm: float


@dataclass
class LOSMotionModel:
    """Continuous-time 3-axis angular LOS trajectory (tip, tilt, roll), in radians.

    Each axis = analytic tonal lines on a colored 1/f**beta floor, normalized to an exact
    target **true RMS** over the seeded buffer grid (``sqrt(mean(theta**2))``, not std — the
    two differ when low-frequency tones have a nonzero finite-window mean). Tones are evaluated
    analytically at any time; the floor is a seeded buffer spanning ``[0, max_duration_s]``
    at ``internal_fs``, sampled by linear interpolation (streaming-safe: index by frame
    timestamp). Sub-window RMS varies around the target; the contract is defined on the full
    buffer. Call :meth:`reset` once per ``perturb`` run, then :meth:`sample`.

    Args:
        tones: ``(frequency_Hz, relative_power_weight)`` lines.
        broadband_beta: Floor exponent (1/f**beta).
        broadband_weight: Fraction of variance in the floor (0..1). With no tone energy,
            a nonzero floor weight is normalized to carry all the energy; with zero floor
            weight there is no motion.
        residual_rms_rad: Per-axis tip/tilt RMS in radians (buffer-grid true RMS).
        roll_ratio: Roll RMS as a fraction of tip/tilt.
        internal_fs: Trajectory sample rate (Hz).
        hf_cutoff_hz: Optional extra low-pass on the floor. The floor is re-standardized after
            filtering so ``broadband_weight`` holds exactly; the cutoff therefore *reshapes*
            the floor (boosting the passband) rather than removing total energy.
        max_duration_s: Length of the seeded floor buffer; sampling beyond raises ValueError.

    Examples:
        >>> import numpy as np
        >>> m = LOSMotionModel(
        ...     tones=[(10.0, 1.0)],
        ...     broadband_beta=2.0,
        ...     broadband_weight=0.1,
        ...     residual_rms_rad=1e-5,
        ...     roll_ratio=1.0,
        ...     internal_fs=1000.0,
        ...     hf_cutoff_hz=None,
        ...     max_duration_s=2.0,
        ... )
        >>> m.reset(rng=np.random.default_rng(0))
        >>> t = np.arange(2001) / 1000.0  # the full buffer grid
        >>> tx, ty, tr = m.sample(t=t)
        >>> bool(abs(np.sqrt((tx**2).mean()) / 1e-5 - 1.0) < 1e-9)
        True
    """

    tones: Sequence[tuple[float, float]]
    broadband_beta: float
    broadband_weight: float
    residual_rms_rad: float
    roll_ratio: float
    internal_fs: float
    hf_cutoff_hz: float | None
    max_duration_s: float
    _t_buf: np.ndarray[Any, Any] = field(init=False, repr=False, default_factory=lambda: np.empty(0))
    _axes: list[_AxisState] = field(init=False, repr=False, default_factory=list)
    _scales: tuple[float, float, float] = field(init=False, repr=False, default=(0.0, 0.0, 0.0))

    def reset(self, *, rng: np.random.Generator) -> None:
        """Draw tone phases and generate the seeded floor buffers for all 3 axes.

        Args:
            rng: Seeded numpy Generator; consumed to draw phases and floor realizations.

        Examples:
            >>> import numpy as np
            >>> m = LOSMotionModel(
            ...     tones=[],
            ...     broadband_beta=2.0,
            ...     broadband_weight=1.0,
            ...     residual_rms_rad=1e-6,
            ...     roll_ratio=1.0,
            ...     internal_fs=100.0,
            ...     hf_cutoff_hz=None,
            ...     max_duration_s=1.0,
            ... )
            >>> m.reset(rng=np.random.default_rng(0))
            >>> m.sample(t=np.zeros(1))[0].shape
            (1,)
        """
        n = int(np.ceil(self.max_duration_s * self.internal_fs)) + 1
        self._t_buf = np.arange(n) / self.internal_fs
        self._axes = [self._build_axis(rng=rng, n=n) for _ in range(3)]
        self._scales = (
            self.residual_rms_rad,
            self.residual_rms_rad,
            self.residual_rms_rad * self.roll_ratio,
        )

    def _build_axis(self, *, rng: np.random.Generator, n: int) -> _AxisState:
        phases = rng.uniform(low=0.0, high=2.0 * np.pi, size=len(self.tones))
        floor = colored_noise(n=n, beta=self.broadband_beta, fs=self.internal_fs, rng=rng)
        if self.hf_cutoff_hz is not None:
            sos = signal.butter(N=2, Wn=self.hf_cutoff_hz, btype="low", fs=self.internal_fs, output="sos")
            floor = signal.sosfiltfilt(sos=sos, x=floor)
            floor = (floor - floor.mean()) / (floor.std() + 1e-12)
        combined = self._combine(phases=phases, floor_unit=floor, t=self._t_buf)
        rms = float(np.sqrt((combined**2).mean()))
        return _AxisState(phases=phases, floor_unit=floor, norm=1.0 / (rms + 1e-12))

    def _tone_unit(self, *, phases: np.ndarray[Any, Any], t: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if len(self.tones) == 0:
            return np.zeros_like(t)
        total_w = sum(w for _, w in self.tones)
        out = np.zeros_like(t)
        for (freq, w), phase in zip(self.tones, phases, strict=True):
            out += np.sqrt(2.0 * w) * np.sin(2.0 * np.pi * freq * t + phase)
        return out / (np.sqrt(total_w) + 1e-12)

    def _combine(
        self,
        *,
        phases: np.ndarray[Any, Any],
        floor_unit: np.ndarray[Any, Any],
        t: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        tone = self._tone_unit(phases=phases, t=t)
        floor = floor_unit if t is self._t_buf else np.interp(t, self._t_buf, floor_unit)
        bw = self.broadband_weight
        return np.sqrt(1.0 - bw) * tone + np.sqrt(bw) * floor

    def sample(
        self,
        *,
        t: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Return the trajectory sampled at the requested times.

        Args:
            t: Sample times in seconds, all within ``[0, max_duration_s]``.

        Returns:
            Tuple ``(theta_x, theta_y, theta_roll)`` of arrays in radians, same shape as ``t``.

        Raises:
            ValueError: If any ``t`` is negative or exceeds ``max_duration_s`` (the floor
                buffer would otherwise silently freeze at its edge value).

        Examples:
            >>> import numpy as np
            >>> m = LOSMotionModel(
            ...     tones=[],
            ...     broadband_beta=2.0,
            ...     broadband_weight=1.0,
            ...     residual_rms_rad=1e-6,
            ...     roll_ratio=0.0,
            ...     internal_fs=100.0,
            ...     hf_cutoff_hz=None,
            ...     max_duration_s=1.0,
            ... )
            >>> m.reset(rng=np.random.default_rng(0))
            >>> tx, ty, tr = m.sample(t=np.array([0.0, 0.5]))
            >>> bool(np.all(tr == 0.0))  # roll_ratio=0 -> zero roll axis
            True
        """
        if not float(t.min()) >= 0.0:
            raise ValueError("sample times must be non-negative (re-zero your frame timestamps)")
        if float(t.max()) > self.max_duration_s:
            raise ValueError(
                f"sample time {float(t.max()):.3f}s exceeds max_duration_s={self.max_duration_s}; "
                f"raise max_duration_s or re-zero frame timestamps",
            )
        out: list[np.ndarray[Any, Any]] = []
        for state, scale in zip(self._axes, self._scales, strict=True):
            raw = self._combine(phases=state.phases, floor_unit=state.floor_unit, t=t)
            out.append(scale * state.norm * raw)
        return out[0], out[1], out[2]
