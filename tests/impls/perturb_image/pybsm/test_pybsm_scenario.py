import pytest
from pybsm.simulation.scenario import Scenario
from contextlib import nullcontext as does_not_raise
from typing import ContextManager
from smqtk_core.configuration import configuration_test_helper
from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario


@pytest.mark.parametrize("name",
                         ["clear", "cloudy", "hurricane"])
def test_scenario_string_rep(name: str) -> None:
    ihaze = 1
    altitude = 2
    groundRange = 0
    print(name)
    scenario = PybsmScenario(name, ihaze, altitude, groundRange)
    assert name == str(scenario)


def test_scenario_call() -> None:
    ihaze = 1
    altitude = 2
    groundRange = 0
    name = "test"
    scenario = PybsmScenario(name, ihaze, altitude, groundRange)
    assert isinstance(scenario(), Scenario)


@pytest.mark.parametrize("ihaze, altitude, groundRange, name, expectation", [
    (1, 2., 0., "test", does_not_raise()),
    (3, 2., 0., "bad-ihaze", pytest.raises(ValueError, match=r"Invalid ihaze value")),
    (1, 101.3, 0., "bad-alt", pytest.raises(ValueError, match=r"Invalid altitude value")),
    (2, 2., -1.2, "bad-groundrange", pytest.raises(ValueError, match=r"Invalid ground range value"))
])
def test_verify_parameters(
    ihaze: int,
    altitude: float,
    groundRange: float,
    name: str,
    expectation: ContextManager
) -> None:

    with expectation:
        pybsm_scenario = PybsmScenario(name, ihaze, altitude, groundRange)

        # testing PybsmScenario call
        assert pybsm_scenario().ihaze == ihaze
        assert pybsm_scenario().altitude == altitude
        assert pybsm_scenario().name == name
        assert pybsm_scenario().ground_range == groundRange

        # testing PybsmScenario.create_scenario directly
        assert pybsm_scenario.create_scenario().ihaze == ihaze
        assert pybsm_scenario.create_scenario().altitude == altitude
        assert pybsm_scenario.create_scenario().name == name
        assert pybsm_scenario.create_scenario().ground_range == groundRange


def test_config() -> None:
    """
    Test configuration stability.
    """
    ihaze = 1
    altitude = 2
    groundRange = 0
    name = "test"
    inst = PybsmScenario(name, ihaze, altitude, groundRange)
    for i in configuration_test_helper(inst):
        assert i.name == inst.name
        assert i.ihaze == inst.ihaze
        assert i.altitude == inst.altitude
        assert i.groundRange == inst.groundRange
        assert i.aircraftSpeed == inst.aircraftSpeed
        assert i.targetReflectance == inst.targetReflectance
        assert i.targetTemperature == inst.targetTemperature
        assert i.backgroundReflectance == inst.backgroundReflectance
        assert i.backgroundTemperature == inst.backgroundTemperature
        assert i.haWindspeed == inst.haWindspeed
        assert i.cn2at1m == inst.cn2at1m
