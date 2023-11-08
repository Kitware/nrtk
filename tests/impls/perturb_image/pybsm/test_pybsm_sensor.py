import numpy as np
import pytest
import pybsm
from smqtk_core.configuration import configuration_test_helper
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor


@pytest.mark.parametrize("name",
                         ["spatial", "spectra", "hyperspectral"])
def test_sensor_string_rep(name: str) -> None:
    D = 0
    f = 0
    px = 0
    optTransWavelengths = np.array([0.58-.08, 0.58+.08])*1.0e-6
    sensor = PybsmSensor(name, D, f, px, optTransWavelengths)
    assert name == str(sensor)


def test_sensor_call() -> None:
    D = 0
    f = 0
    px = 0
    optTransWavelengths = np.array([0.58-.08, 0.58+.08])*1.0e-6
    name = "test"
    sensor = PybsmSensor(name, D, f, px, optTransWavelengths)
    assert type(sensor()) == pybsm.sensor


def test_verify_parameters() -> None:
    D = 0
    f = 0
    px = 0
    optTransWavelengths = np.array([0.58-.08, 0.58+.08])*1.0e-6
    name = "test"
    sensor = PybsmSensor(name, D, f, px, optTransWavelengths)

    # testing PybsmSensor call
    assert sensor().D == D
    assert sensor().f == f
    assert sensor().px == px
    assert sensor().optTransWavelengths.all() == optTransWavelengths.all()
    assert sensor().name == name

    # testing PybsmSensor.create_sensor directily
    assert sensor.create_sensor().D == D
    assert sensor.create_sensor().f == f
    assert sensor.create_sensor().px == px
    assert sensor.create_sensor().optTransWavelengths.all() == optTransWavelengths.all()
    assert sensor.create_sensor().name == name


def test_config() -> None:
    """
    Test configuration stability.
    """
    D = 0
    f = 0
    px = 0
    optTransWavelengths = np.array([0.58-.08, 0.58+.08])*1.0e-6
    name = "test"
    sensor = PybsmSensor(name, D, f, px, optTransWavelengths)
    configuration_test_helper(sensor)
