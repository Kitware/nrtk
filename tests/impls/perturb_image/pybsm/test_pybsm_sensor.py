import numpy as np
import pytest
from pybsm.simulation.sensor import Sensor
from contextlib import nullcontext as does_not_raise
from typing import ContextManager, Optional
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
    px = 0.1
    optTransWavelengths = np.array([0.58-.08, 0.58+.08])*1.0e-6
    name = "test"
    sensor = PybsmSensor(name, D, f, px, optTransWavelengths)
    assert isinstance(sensor(), Sensor)


@pytest.mark.parametrize("optTransWavelengths, opticsTransmission, name, expectation", [
    (np.array([0.58-.08, 0.58+.08])*1.0e-6, None, "test", does_not_raise()),
    (np.array([0.1]), None, "not-enough-wavelengths", pytest.raises(ValueError, match=r"At minimum, at least")),
    (np.array([5, 0.5]), None, "descending",
        pytest.raises(ValueError, match=r"optTransWavelengths must be ascending")),
    (np.array([0.5, 1.]), np.array([0.5]), "mismatched-sizes",
        pytest.raises(ValueError, match=r"opticsTransmission and optTransWavelengths must have the same length"))
])
def test_verify_parameters(
    optTransWavelengths: np.ndarray,
    opticsTransmission: Optional[np.ndarray],
    name: str,
    expectation: ContextManager
) -> None:
    D = 0
    f = 0
    px = 0.1
    with expectation:
        sensor = PybsmSensor(name, D, f, px, optTransWavelengths, opticsTransmission=opticsTransmission)

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
