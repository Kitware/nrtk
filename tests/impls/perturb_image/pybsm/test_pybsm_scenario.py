import pytest
import pybsm
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
    assert type(scenario()) == pybsm.scenario


def test_verify_parameters() -> None:
    ihaze = 1
    altitude = 2
    groundRange = 0
    name = "test"
    scenario = PybsmScenario(name, ihaze, altitude, groundRange)

    # testing PybsmScenario call
    assert scenario().ihaze == ihaze
    assert scenario().altitude == altitude
    assert scenario().name == name
    assert scenario().groundRange == groundRange

    # testing PybsmScenario.create_scenario directly
    assert scenario.create_scenario().ihaze == ihaze
    assert scenario.create_scenario().altitude == altitude
    assert scenario.create_scenario().name == name
    assert scenario.create_scenario().groundRange == groundRange


def test_config() -> None:
    """
    Test configuration stability.
    """
    ihaze = 1
    altitude = 2
    groundRange = 0
    name = "test"
    scenario = PybsmScenario(name, ihaze, altitude, groundRange)
    configuration_test_helper(scenario)
