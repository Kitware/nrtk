"""Tests for load_default_config configuration loading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import nrtk.impls.perturb_image.optical._pybsm._default_config as _mod
from nrtk.impls.perturb_image.optical.otf import load_default_config

_DATA_DIR = Path(_mod.__file__).resolve().parent / "_data"


@pytest.mark.pybsm
class TestDefaultConfig:
    """Tests for load_default_config."""

    @pytest.mark.parametrize("preset", ["blackfly", "sample"])
    def test_preset_loadable(self, preset: str) -> None:
        """All bundled presets should load successfully."""
        result = load_default_config(preset=preset)

        assert "sensor_name" in result
        assert "scenario_name" in result

    def test_load_with_explicit_data_dir(self) -> None:
        """Load config from an explicit data_dir and verify numpy conversion."""
        result = load_default_config(preset="blackfly", data_dir=str(_DATA_DIR))

        assert isinstance(result["opt_trans_wavelengths"], np.ndarray)
        assert isinstance(result["qe_wavelengths"], np.ndarray)
        assert isinstance(result["qe"], np.ndarray)

    def test_unsupported_preset(self) -> None:
        """ValueError for an unknown preset."""
        with pytest.raises(ValueError, match="Unsupported configuration type"):
            load_default_config(preset="satellite")

    def test_bad_data_dir(self, tmp_path: Path) -> None:
        """Errors when data_dir doesn't exist or doesn't contain the config file."""
        with pytest.raises(ValueError, match="Specified data directory does not exist"):
            load_default_config(preset="blackfly", data_dir=str(tmp_path / "does_not_exist"))

        with pytest.raises(FileNotFoundError, match="not found in specified data directory"):
            load_default_config(preset="blackfly", data_dir=str(tmp_path))

    def test_missing_required_keys(self, tmp_path: Path) -> None:
        """ValueError when JSON is valid but missing 'sensor' or 'scenario' keys."""
        (tmp_path / "blackfly.json").write_text(json.dumps({"sensor": {"sensor_name": "test"}}))

        with pytest.raises(ValueError, match="missing required keys"):
            load_default_config(preset="blackfly", data_dir=str(tmp_path))

    def test_invalid_json(self, tmp_path: Path) -> None:
        """JSONDecodeError when config file contains invalid JSON."""
        (tmp_path / "blackfly.json").write_text("not valid json {{{")

        with pytest.raises(json.JSONDecodeError):
            load_default_config(preset="blackfly", data_dir=str(tmp_path))

    def test_fallback_paths_not_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """FileNotFoundError when no data_dir given and fallback paths don't exist."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(_mod, "__file__", str(tmp_path / "_default_config.py"))  # noqa: FKA100

        with pytest.raises(FileNotFoundError, match="not found. Searched paths"):
            load_default_config(preset="blackfly")
