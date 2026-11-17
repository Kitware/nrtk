"""Tests for the LOS motion model (colored noise, tone comb, trajectory)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from nrtk.impls.perturb_video.optical._jitter.los_motion import (
    LOSMotionModel,
    colored_noise,
    rotating_element_comb,
)


@pytest.mark.scipy
def test_colored_noise_is_unit_variance_and_reproducible() -> None:
    """Same seed gives identical output; result is zero-mean with unit standard deviation."""
    x1 = colored_noise(n=8192, beta=2.0, fs=1000.0, rng=np.random.default_rng(0))
    x2 = colored_noise(n=8192, beta=2.0, fs=1000.0, rng=np.random.default_rng(0))
    assert x1.shape == (8192,)
    np.testing.assert_array_equal(x1, x2)  # same seed -> identical
    assert abs(x1.std() - 1.0) < 1e-9  # unit std (building block; exact RMS is the model's job)
    assert abs(x1.mean()) < 1e-9  # zero mean


@pytest.mark.scipy
def test_colored_noise_slope_steeper_for_higher_beta() -> None:
    """A larger beta concentrates more spectral power at low frequencies."""
    fs, n = 1000.0, 1 << 15

    def band_ratio(x: np.ndarray) -> float:
        f, p = signal.welch(x, fs=fs, nperseg=4096)
        lo = p[(f >= 1.0) & (f < 5.0)].mean()
        hi = p[(f >= 50.0) & (f < 200.0)].mean()
        return float(lo / hi)

    r2 = band_ratio(colored_noise(n=n, beta=2.0, fs=fs, rng=np.random.default_rng(1)))
    r1 = band_ratio(colored_noise(n=n, beta=1.0, fs=fs, rng=np.random.default_rng(1)))
    assert r2 > r1  # beta=2 is redder than beta=1


@pytest.mark.scipy
def test_rotating_element_comb_blade_pass_and_rolloff() -> None:
    """The comb places lines at 1P and blade-pass harmonics with decaying weights."""
    # 4-blade rotor at 290 rpm -> 1P = 4.833 Hz, blade-pass = 19.33 Hz
    tones = rotating_element_comb(fundamental_hz=290.0 / 60.0, n_blades=4, n_harmonics=3)
    freqs = [f for f, _ in tones]
    weights = [w for _, w in tones]
    assert freqs[0] == pytest.approx(4.8333, abs=1e-3)  # 1P
    assert freqs[1] == pytest.approx(19.333, abs=1e-3)  # blade-pass = 4 * 1P
    assert weights[1] > weights[2] > weights[3]  # harmonics decay


def _model(**kw: object) -> LOSMotionModel:
    """Build a LOSMotionModel with small test defaults, overridable per test."""
    base: dict[str, object] = {
        "tones": [(20.0, 1.0)],
        "broadband_beta": 2.0,
        "broadband_weight": 0.1,
        "residual_rms_rad": 15e-6,
        "roll_ratio": 1.0,
        "internal_fs": 2000.0,
        "hf_cutoff_hz": None,
        "max_duration_s": 10.0,
    }
    base.update(kw)
    return LOSMotionModel(**base)  # type: ignore[arg-type]


@pytest.mark.scipy
def test_model_exact_rms_on_buffer_grid() -> None:
    """True RMS over the buffer grid equals the target exactly; roll scales by roll_ratio."""
    m = _model(roll_ratio=0.5)
    m.reset(rng=np.random.default_rng(0))
    t = np.arange(int(round(10.0 * 2000.0)) + 1) / 2000.0  # exactly the buffer grid
    tx, ty, troll = m.sample(t=t)
    assert tx.shape == ty.shape == troll.shape == t.shape
    assert np.sqrt((tx**2).mean()) == pytest.approx(15e-6, rel=1e-9)  # exact TRUE RMS
    assert np.sqrt((ty**2).mean()) == pytest.approx(15e-6, rel=1e-9)
    assert np.sqrt((troll**2).mean()) == pytest.approx(0.5 * 15e-6, rel=1e-9)  # roll = ratio * tip


@pytest.mark.scipy
def test_model_exact_rms_with_subclip_length_tones() -> None:
    """The RMS contract holds even when slow tones make std and RMS diverge.

    A 0.09 Hz tone on a 10 s buffer has a large finite-window mean, so std and RMS differ by
    tens of percent; the normalization must target true RMS exactly.
    """
    m = _model(tones=[(0.09, 1.0), (0.13, 0.55)], broadband_weight=0.06)
    m.reset(rng=np.random.default_rng(3))
    t = np.arange(int(round(10.0 * 2000.0)) + 1) / 2000.0
    tx, _, _ = m.sample(t=t)
    assert np.sqrt((tx**2).mean()) == pytest.approx(15e-6, rel=1e-9)


@pytest.mark.scipy
def test_model_reproducible_and_axes_independent() -> None:
    """Same seed reproduces the trajectory bit-for-bit; x and y axes are distinct draws.

    Axis independence must be asserted on a floor-dominated config: two equal-frequency
    tones have Pearson correlation cos(phase difference), which is arbitrary by design.
    """
    m1, m2 = _model(), _model()
    m1.reset(rng=np.random.default_rng(7))
    m2.reset(rng=np.random.default_rng(7))
    t = np.linspace(0.0, 5.0, 5000)
    a1, a2 = m1.sample(t=t), m2.sample(t=t)
    for x, y in zip(a1, a2, strict=True):
        np.testing.assert_array_equal(x, y)  # same seed -> identical
    m3 = _model(tones=[], broadband_beta=1.0, broadband_weight=1.0)
    m3.reset(rng=np.random.default_rng(7))
    fx, fy, _ = m3.sample(t=t)
    assert abs(np.corrcoef(fx, fy)[0, 1]) < 0.3  # x and y are distinct draws


@pytest.mark.scipy
def test_model_spectral_peak_at_tone() -> None:
    """The generated trajectory's dominant spectral line sits at the requested tone."""
    m = _model(tones=[(20.0, 1.0)], broadband_weight=0.05)
    m.reset(rng=np.random.default_rng(0))
    t = np.arange(0.0, 10.0, 1.0 / 2000.0)
    tx, _, _ = m.sample(t=t)
    f, p = signal.welch(tx, fs=2000.0, nperseg=4096)
    assert f[np.argmax(p)] == pytest.approx(20.0, abs=1.0)  # dominant line at 20 Hz


@pytest.mark.scipy
def test_model_hf_cutoff_attenuates_high_band() -> None:
    """hf_cutoff_hz suppresses floor power above the cutoff."""

    def hi_power(cut: float | None) -> float:
        m = _model(tones=[], broadband_weight=1.0, hf_cutoff_hz=cut)
        m.reset(rng=np.random.default_rng(0))
        t = np.arange(0.0, 10.0, 1.0 / 2000.0)
        tx, _, _ = m.sample(t=t)
        f, p = signal.welch(tx, fs=2000.0, nperseg=4096)
        return float(p[(f > 200.0)].sum())

    assert hi_power(50.0) < 0.1 * hi_power(None)


@pytest.mark.scipy
def test_model_sample_out_of_range_raises() -> None:
    """Sampling outside [0, max_duration_s] raises instead of silently clamping the floor."""
    m = _model(max_duration_s=2.0)
    m.reset(rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="max_duration_s"):
        m.sample(t=np.array([1.0, 2.5]))  # beyond the buffer -> loud failure, never a frozen clamp
    with pytest.raises(ValueError, match="non-negative"):
        m.sample(t=np.array([-0.1, 1.0]))
