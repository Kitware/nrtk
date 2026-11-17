"""Defines the per-platform jitter preset registry for the FMV jitter perturber.

Classes:
    PresetSpec: One platform preset: source spectrum, optics defaults, and provenance.

Functions:
    load_preset: Load one preset from its JSON file in ``_data/``.
    list_platforms: Platform names available to ``JitterVideoPerturber.from_platform``.
    resolve_preset: Resolve a platform (+ optional sub-mode) to its ``PresetSpec``.

Preset values live in ``_data/<name>.json``, one file per preset in the same style as the pyBSM
sensor/scenario presets, and are loaded through :func:`load_preset`. ``PRESET_FILES`` is the
single mapping of preset names to files; both preset resolution and platform listing read it.

Tonal FREQUENCIES/structure are citable (MIL-STD-810H Method 514.8, MIL-STD-167-1A, and the
papers named per preset); every residual-RMS AMPLITUDE is a representative engineering default
(no unclassified measured residual-LOS PSD exists for fielded EO/IR turrets) — hence the
per-preset ``shape_provisional`` flag, while amplitudes are representative across the board.
Citations are system-agnostic by policy (standards and papers only). Where a standard tabulates
its parameters by example aircraft (MIL-STD-810H helicopter and propeller tables), a preset uses
one published example row without naming the platform.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_MODES",
    "MODE_NAMES",
    "PRESET_FILES",
    "PresetSpec",
    "list_platforms",
    "load_preset",
    "resolve_preset",
]

import json
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).parent / "_data"

# ``residual_rms`` is authored in the file's ``units`` and converted to radians on load by
# dividing by the units per radian: dividing (rather than multiplying by the reciprocal)
# keeps a decimal-authored 15 urad bit-identical to the literal 15e-6.
_UNITS_PER_RAD: dict[str, float] = {
    "urad": 1e6,
    "mas": 1.0 / 4.848e-9,  # 4.848e-9 radians per milliarcsecond
}

PRESET_FILES: dict[str, str] = {
    "helicopter": "helicopter.json",
    "fixed_wing": "fixed_wing.json",
    "multirotor_uas": "multirotor_uas.json",
    "ground_tracked": "ground_tracked.json",
    "ground_wheeled": "ground_wheeled.json",
    "maritime_coarse": "maritime_coarse.json",
    "maritime_fine": "maritime_fine.json",
    "handheld_unstabilized": "handheld_unstabilized.json",
    "handheld_ois": "handheld_ois.json",
    "satellite": "satellite.json",
}

DEFAULT_MODES: dict[str, str] = {"maritime": "coarse", "handheld": "unstabilized"}
MODE_NAMES: tuple[str, ...] = ("fine", "coarse", "unstabilized", "ois")


@dataclass(frozen=True)
class PresetSpec:
    """One platform preset: source spectrum + optics defaults + provenance."""

    tones: tuple[tuple[float, float], ...]
    residual_rms: float  # radians, per-axis tip/tilt true RMS
    pixel_pitch: float
    focal_length: float
    integration_time: float
    broadband_beta: float = 2.0
    broadband_weight: float = 0.1
    roll_ratio: float = 1.0
    rolling_shutter_readout_s: float | None = None
    units: str = "urad"
    citation: str = ""
    shape_provisional: bool = True
    notes: str = ""


_REQUIRED_KEYS: tuple[str, ...] = tuple(f.name for f in fields(PresetSpec) if f.default is MISSING)
_KNOWN_KEYS: frozenset[str] = frozenset(f.name for f in fields(PresetSpec))


def _preset_dir(*, data_dir: str | None) -> Path:
    """Return the directory holding the preset files; the bundled ``_data/`` when ``data_dir`` is None."""
    if data_dir is None:
        return _DATA_DIR
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise ValueError(f"Specified data directory does not exist: {data_dir}")
    return data_path


def _preset_path(*, preset: str, data_dir: str | None) -> Path:
    """Resolve a preset name to its file through ``PRESET_FILES``."""
    if preset not in PRESET_FILES:
        available = ", ".join(PRESET_FILES)
        raise ValueError(f"Unsupported preset: '{preset}'. Available presets: {available}")
    directory = _preset_dir(data_dir=data_dir)
    preset_path = directory / PRESET_FILES[preset]
    if not preset_path.exists():
        where = "bundled" if data_dir is None else "specified"
        raise FileNotFoundError(f"Preset file '{preset_path.name}' not found in {where} data directory: {directory}")
    return preset_path


def _read_preset_file(*, preset_path: Path) -> dict[str, Any]:
    """Parse a preset file, naming the file in any JSON or I/O error."""
    try:
        with open(preset_path, encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in preset file '{preset_path}': {e}", e.doc, e.pos) from e
    except OSError as e:
        raise OSError(f"Error reading preset file '{preset_path}': {e}") from e


def _spec_from_data(*, preset_path: Path, preset_data: dict[str, Any]) -> PresetSpec:
    """Validate the parsed keys and build the ``PresetSpec``, converting ``residual_rms`` to radians."""
    missing_keys = [key for key in _REQUIRED_KEYS if key not in preset_data]
    if missing_keys:
        raise ValueError(f"Preset file '{preset_path}' is missing required keys: {missing_keys}")
    unexpected_keys = sorted(set(preset_data) - _KNOWN_KEYS)
    if unexpected_keys:
        raise ValueError(f"Preset file '{preset_path}' has unexpected keys: {unexpected_keys}")
    units = preset_data.get("units", PresetSpec.units)
    if units not in _UNITS_PER_RAD:
        raise ValueError(
            f"Preset file '{preset_path}' has unknown units '{units}'; expected one of {sorted(_UNITS_PER_RAD)}",
        )
    preset_data["tones"] = tuple((float(f), float(w)) for f, w in preset_data["tones"])
    preset_data["residual_rms"] = float(preset_data["residual_rms"]) / _UNITS_PER_RAD[units]
    return PresetSpec(**preset_data)


def load_preset(*, preset: str, data_dir: str | None = None) -> PresetSpec:
    """Load one preset from its JSON file.

    The file holds the ``PresetSpec`` fields as JSON keys. ``residual_rms`` is authored in
    the file's ``units`` (``urad`` or ``mas``) and converted to radians here, so the returned
    spec is always in SI.

    Args:
        preset:
            Preset name; one of the keys of ``PRESET_FILES``.
        data_dir:
            Optional directory holding the preset files. ``None`` reads the bundled
            ``_data/`` directory next to this module.

    Returns:
        The loaded ``PresetSpec``.

    Raises:
        ValueError:
            If the preset name is not supported, ``data_dir`` does not exist, or the file is
            missing required keys, holds unexpected keys, or names unknown units.
        FileNotFoundError:
            If the preset file cannot be found.
        json.JSONDecodeError:
            If the preset file contains invalid JSON.
        OSError:
            If there is an error reading the preset file.

    Examples:
        >>> load_preset(preset="helicopter").shape_provisional
        False
    """
    preset_path = _preset_path(preset=preset, data_dir=data_dir)
    return _spec_from_data(preset_path=preset_path, preset_data=_read_preset_file(preset_path=preset_path))


def list_platforms() -> list[str]:
    """Return the available preset platform names (mode-bearing platforms listed once).

    Examples:
        >>> "helicopter" in list_platforms()
        True
    """
    moded = {f"{p}_{m}" for p in DEFAULT_MODES for m in MODE_NAMES}
    return sorted({k for k in PRESET_FILES if k not in moded} | set(DEFAULT_MODES))


def resolve_preset(*, platform: str, mode: str | None) -> PresetSpec:
    """Resolve a platform (+ optional sub-mode) to its PresetSpec.

    Args:
        platform: Platform name; ``maritime`` and ``handheld`` accept a ``mode``.
        mode: Sub-mode for platforms that define one; None selects the platform default.

    Returns:
        The matching PresetSpec.

    Raises:
        ValueError: Unknown platform, unknown mode, or a mode passed for a mode-less platform.

    Examples:
        >>> resolve_preset(platform="helicopter", mode=None).shape_provisional
        False
    """
    if platform in DEFAULT_MODES:
        key = f"{platform}_{mode or DEFAULT_MODES[platform]}"
        if key not in PRESET_FILES:
            raise ValueError(f"unknown mode {mode!r} for platform {platform!r}")
        return load_preset(preset=key)
    if platform not in PRESET_FILES:
        raise ValueError(f"unknown platform {platform!r}; choose from {list_platforms()}")
    if mode is not None:
        raise ValueError(f"platform {platform!r} has no mode variants (got mode={mode!r})")
    return load_preset(preset=platform)
