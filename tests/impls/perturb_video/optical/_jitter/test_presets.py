"""Tests for the platform preset registry: JSON loading, resolution, and the provenance surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import nrtk.impls.perturb_video.optical._jitter._presets as _mod
from nrtk.impls.perturb_video.optical import JitterVideoPerturber
from nrtk.impls.perturb_video.optical._jitter._presets import PRESET_FILES, PresetSpec, load_preset

_DATA_DIR = Path(_mod.__file__).resolve().parent / "_data"


@pytest.mark.scipy
class TestLoadPreset:
    """Tests for load_preset."""

    @pytest.mark.parametrize("preset", sorted(PRESET_FILES))
    def test_preset_loadable(self, preset: str) -> None:
        """All bundled presets should load into a fully populated, in-range PresetSpec."""
        spec = load_preset(preset=preset)

        assert isinstance(spec, PresetSpec)
        assert isinstance(spec.tones, tuple)
        assert all(isinstance(f, float) and isinstance(w, float) for f, w in spec.tones)
        assert all(f > 0.0 and w >= 0.0 for f, w in spec.tones)
        assert 0.0 < spec.residual_rms < 1e-2  # radians, not the authored urad/mas figure
        assert spec.pixel_pitch > 0.0
        assert spec.focal_length > 0.0
        assert spec.integration_time > 0.0
        assert spec.broadband_beta >= 0.0
        assert 0.0 <= spec.broadband_weight <= 1.0
        assert spec.roll_ratio >= 0.0
        assert spec.rolling_shutter_readout_s is None or spec.rolling_shutter_readout_s > 0.0
        assert spec.citation

    def test_every_bundled_file_is_registered(self) -> None:
        """The mapping of names to files and the files on disk agree exactly."""
        assert {p.name for p in _DATA_DIR.glob("*.json")} == set(PRESET_FILES.values())

    def test_load_with_explicit_data_dir(self) -> None:
        """Load a preset from an explicit data_dir and verify it matches the bundled one."""
        result = load_preset(preset="helicopter", data_dir=str(_DATA_DIR))

        assert result == load_preset(preset="helicopter")

    def test_units_converted_to_radians(self, tmp_path: Path) -> None:
        """residual_rms is authored in the file's units and returned in radians."""
        data = json.loads((_DATA_DIR / "helicopter.json").read_text())
        data["residual_rms"] = 2.0
        data["units"] = "urad"
        (tmp_path / "helicopter.json").write_text(json.dumps(data))
        assert load_preset(preset="helicopter", data_dir=str(tmp_path)).residual_rms == 2e-6

        data["units"] = "mas"
        (tmp_path / "helicopter.json").write_text(json.dumps(data))
        assert load_preset(preset="helicopter", data_dir=str(tmp_path)).residual_rms == pytest.approx(
            2.0 * 4.848e-9,
            rel=1e-9,
        )

    def test_unsupported_preset(self) -> None:
        """ValueError for an unknown preset."""
        with pytest.raises(ValueError, match="Unsupported preset"):
            load_preset(preset="spaceship")

    def test_bad_data_dir(self, tmp_path: Path) -> None:
        """Errors when data_dir doesn't exist or doesn't contain the preset file."""
        with pytest.raises(ValueError, match="Specified data directory does not exist"):
            load_preset(preset="helicopter", data_dir=str(tmp_path / "does_not_exist"))

        with pytest.raises(FileNotFoundError, match="not found in specified data directory"):
            load_preset(preset="helicopter", data_dir=str(tmp_path))

    def test_missing_required_keys(self, tmp_path: Path) -> None:
        """ValueError when JSON is valid but missing a required PresetSpec field."""
        (tmp_path / "helicopter.json").write_text(json.dumps({"tones": [[20.0, 1.0]], "residual_rms": 15.0}))

        with pytest.raises(ValueError, match="missing required keys"):
            load_preset(preset="helicopter", data_dir=str(tmp_path))

    def test_unexpected_keys(self, tmp_path: Path) -> None:
        """ValueError when JSON holds a key that is not a PresetSpec field."""
        data = json.loads((_DATA_DIR / "helicopter.json").read_text())
        data["platform"] = "helicopter"
        (tmp_path / "helicopter.json").write_text(json.dumps(data))

        with pytest.raises(ValueError, match="unexpected keys"):
            load_preset(preset="helicopter", data_dir=str(tmp_path))

    def test_unknown_units(self, tmp_path: Path) -> None:
        """ValueError when the amplitude units are not convertible to radians."""
        data = json.loads((_DATA_DIR / "helicopter.json").read_text())
        data["units"] = "furlongs"
        (tmp_path / "helicopter.json").write_text(json.dumps(data))

        with pytest.raises(ValueError, match="unknown units"):
            load_preset(preset="helicopter", data_dir=str(tmp_path))

    def test_invalid_json(self, tmp_path: Path) -> None:
        """JSONDecodeError when the preset file contains invalid JSON."""
        (tmp_path / "helicopter.json").write_text("not valid json {{{")

        with pytest.raises(json.JSONDecodeError):
            load_preset(preset="helicopter", data_dir=str(tmp_path))

    def test_bundled_file_not_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """FileNotFoundError when no data_dir is given and the bundled file is missing."""
        monkeypatch.setattr(_mod, "_DATA_DIR", tmp_path / "missing")  # noqa: FKA100

        with pytest.raises(FileNotFoundError, match="not found in bundled data directory"):
            load_preset(preset="helicopter")


@pytest.mark.scipy
def test_list_platforms_includes_core_set() -> None:
    """The registry exposes all eight core platform names."""
    names = set(JitterVideoPerturber.list_platforms())
    assert {
        "helicopter",
        "fixed_wing",
        "multirotor_uas",
        "ground_tracked",
        "ground_wheeled",
        "maritime",
        "handheld",
        "satellite",
    } <= names


@pytest.mark.scipy
def test_list_platforms_and_resolution_share_the_file_mapping() -> None:
    """Every listed platform resolves, and every registered file is reachable from the listing."""
    reached: set[str] = set()
    for name in JitterVideoPerturber.list_platforms():
        if name in _mod.DEFAULT_MODES:
            for mode in _mod.MODE_NAMES:
                key = f"{name}_{mode}"
                if key in PRESET_FILES:
                    assert JitterVideoPerturber.from_platform(name, mode=mode).tones
                    reached.add(key)
        else:
            assert JitterVideoPerturber.from_platform(name).tones
            reached.add(name)
    assert reached == set(PRESET_FILES)


@pytest.mark.scipy
def test_every_preset_constructs_and_round_trips() -> None:
    """Every preset builds a perturber whose config survives from_config round-trips."""
    for name in JitterVideoPerturber.list_platforms():
        p = JitterVideoPerturber.from_platform(name, seed=1)
        assert JitterVideoPerturber.from_config(p.get_config()).get_config() == p.get_config()


@pytest.mark.scipy
def test_from_platform_overrides_and_severity() -> None:
    """Severity and keyword overrides take precedence over preset defaults."""
    p = JitterVideoPerturber.from_platform("helicopter", severity=2.0, seed=5)
    assert p.severity == 2.0
    assert len(p.tones) > 0
    q = JitterVideoPerturber.from_platform("helicopter", integration_time=4e-3)
    assert q.integration_time == 4e-3


@pytest.mark.scipy
def test_sub_modes_resolve() -> None:
    """Maritime and handheld sub-modes resolve; mode-less platforms reject a mode."""
    fine = JitterVideoPerturber.from_platform("maritime", mode="fine")
    coarse = JitterVideoPerturber.from_platform("maritime")  # default mode = coarse
    assert fine.residual_rms < coarse.residual_rms
    ois = JitterVideoPerturber.from_platform("handheld", mode="ois")
    raw = JitterVideoPerturber.from_platform("handheld")  # default = unstabilized
    assert ois.residual_rms < raw.residual_rms
    with pytest.raises(ValueError, match="mode"):
        JitterVideoPerturber.from_platform("helicopter", mode="fine")  # no modes on helicopter
    with pytest.raises(ValueError, match="unknown mode"):
        JitterVideoPerturber.from_platform("maritime", mode="turbo")  # moded platform, bad mode


@pytest.mark.scipy
def test_unknown_platform_raises() -> None:
    """An unknown platform name raises with the valid choices in the message."""
    with pytest.raises(ValueError, match="unknown platform"):
        JitterVideoPerturber.from_platform("spaceship")


@pytest.mark.scipy
def test_multirotor_defaults_to_rolling_shutter() -> None:
    """The multirotor preset enables rolling shutter by default (CMOS sensor class)."""
    assert JitterVideoPerturber.from_platform("multirotor_uas").rolling_shutter_readout_s == 16e-3


@pytest.mark.scipy
def test_provenance_keys_and_flags() -> None:
    """platform_provenance returns the full audit record with correct provisional flags."""
    prov = JitterVideoPerturber.platform_provenance("helicopter")
    assert set(prov) == {
        "citation",
        "shape_provisional",
        "units",
        "default_residual_rms",
        "notes",
    }
    assert prov["shape_provisional"] is False  # 810H rotor tone formulas
    assert JitterVideoPerturber.platform_provenance("ground_wheeled")["shape_provisional"] is True


@pytest.mark.scipy
def test_satellite_mas_conversion() -> None:
    """The satellite preset is authored in milliarcseconds and converted to radians."""
    prov = JitterVideoPerturber.platform_provenance("satellite")
    assert prov["units"] == "mas"
    assert prov["default_residual_rms"] == pytest.approx(103.0 * 4.848e-9, rel=1e-6)  # ~0.5 urad
