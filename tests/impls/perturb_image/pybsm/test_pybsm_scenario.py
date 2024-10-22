import unittest.mock as mock
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import pytest
from pybsm.simulation.scenario import Scenario
from smqtk_core.configuration import configuration_test_helper

from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario


@pytest.mark.skipif(
    not PybsmScenario.is_usable(),
    reason="pybsm not found. Please install 'nrtk[pybsm]', 'nrtk[pybsm-graphics]', or `nrtk[pybsm-headless]`.",
)
@pytest.mark.parametrize("name", ["clear", "cloudy", "hurricane"])
def test_scenario_string_rep(name: str) -> None:
    ihaze = 1
    altitude = 2
    ground_range = 0
    print(name)
    scenario = PybsmScenario(name, ihaze, altitude, ground_range)
    assert name == str(scenario)


@pytest.mark.skipif(
    not PybsmScenario.is_usable(),
    reason="pybsm not found. Please install 'nrtk[pybsm]', 'nrtk[pybsm-graphics]', or `nrtk[pybsm-headless]`.",
)
def test_scenario_call() -> None:
    ihaze = 1
    altitude = 2
    ground_range = 0
    name = "test"
    scenario = PybsmScenario(name, ihaze, altitude, ground_range)
    assert isinstance(scenario(), Scenario)


@pytest.mark.skipif(
    not PybsmScenario.is_usable(),
    reason="pybsm not found. Please install 'nrtk[pybsm]', 'nrtk[pybsm-graphics]', or `nrtk[pybsm-headless]`.",
)
@pytest.mark.parametrize(
    ("ihaze", "altitude", "ground_range", "name", "expectation"),
    [
        (1, 2.0, 0.0, "test", does_not_raise()),
        (
            3,
            2.0,
            0.0,
            "bad-ihaze",
            pytest.raises(ValueError, match=r"Invalid ihaze value"),
        ),
        (
            1,
            101.3,
            0.0,
            "bad-alt",
            pytest.raises(ValueError, match=r"Invalid altitude value"),
        ),
        (
            2,
            2.0,
            -1.2,
            "bad-ground_range",
            pytest.raises(ValueError, match=r"Invalid ground range value"),
        ),
    ],
)
def test_verify_parameters(
    ihaze: int,
    altitude: float,
    ground_range: float,
    name: str,
    expectation: AbstractContextManager,
) -> None:
    with expectation:
        pybsm_scenario = PybsmScenario(name, ihaze, altitude, ground_range)

        # testing PybsmScenario call
        assert pybsm_scenario().ihaze == ihaze
        assert pybsm_scenario().altitude == altitude
        assert pybsm_scenario().name == name
        assert pybsm_scenario().ground_range == ground_range

        # testing PybsmScenario.create_scenario directly
        assert pybsm_scenario.create_scenario().ihaze == ihaze
        assert pybsm_scenario.create_scenario().altitude == altitude
        assert pybsm_scenario.create_scenario().name == name
        assert pybsm_scenario.create_scenario().ground_range == ground_range


@pytest.mark.skipif(
    not PybsmScenario.is_usable(),
    reason="pybsm not found. Please install 'nrtk[pybsm]', 'nrtk[pybsm-graphics]', or `nrtk[pybsm-headless]`.",
)
def test_config() -> None:
    """Test configuration stability."""
    ihaze = 1
    altitude = 2
    ground_range = 0
    name = "test"
    inst = PybsmScenario(name, ihaze, altitude, ground_range)
    for i in configuration_test_helper(inst):
        assert i.name == inst.name
        assert i.ihaze == inst.ihaze
        assert i.altitude == inst.altitude
        assert i.ground_range == inst.ground_range
        assert i.aircraft_speed == inst.aircraft_speed
        assert i.target_reflectance == inst.target_reflectance
        assert i.target_temperature == inst.target_temperature
        assert i.background_reflectance == inst.background_reflectance
        assert i.background_temperature == inst.background_temperature
        assert i.ha_wind_speed == inst.ha_wind_speed
        assert i.cn2_at_1m == inst.cn2_at_1m


@mock.patch.object(PybsmScenario, "is_usable")
def test_missing_deps(mock_is_usable) -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    mock_is_usable.return_value = False
    assert not PybsmScenario.is_usable()
    name = "clear"
    ihaze = 1
    altitude = 2
    ground_range = 0
    with pytest.raises(ImportError, match=r"pybsm not found"):
        PybsmScenario(name, ihaze, altitude, ground_range)
