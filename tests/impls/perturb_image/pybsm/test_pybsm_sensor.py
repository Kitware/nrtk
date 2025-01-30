from __future__ import annotations

from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from smqtk_core.configuration import configuration_test_helper

try:
    from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
except ImportError:
    pytest.skip(allow_module_level=True, reason="nrtk.impls.perturb_image.pybsm submodule unavailable.")
try:
    from pybsm.simulation.sensor import Sensor
except ImportError:
    pytest.skip(allow_module_level=True, reason="pybsm.simulation.sensor.Sensor unavailable.")


@pytest.mark.parametrize("name", ["spatial", "spectra", "hyperspectral"])
def test_sensor_string_rep(name: str) -> None:
    D = 0  # noqa:N806
    f = 0
    p_x = 0
    opt_trans_wavelengths = np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6
    sensor = PybsmSensor(name, D, f, p_x, opt_trans_wavelengths)
    assert name == str(sensor)


def test_sensor_call() -> None:
    D = 0  # noqa:N806
    f = 0
    p_x = 0.1
    opt_trans_wavelengths = np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6
    name = "test"
    sensor = PybsmSensor(name, D, f, p_x, opt_trans_wavelengths)
    assert isinstance(sensor(), Sensor)


@pytest.mark.parametrize(
    ("opt_trans_wavelengths", "optics_transmission", "name", "expectation"),
    [
        (np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6, None, "test", does_not_raise()),
        (
            np.array([0.1]),
            None,
            "not-enough-wavelengths",
            pytest.raises(ValueError, match=r"At minimum, at least"),
        ),
        (
            np.array([5, 0.5]),
            None,
            "descending",
            pytest.raises(ValueError, match=r"opt_trans_wavelengths must be ascending"),
        ),
        (
            np.array([0.5, 1.0]),
            np.array([0.5]),
            "mismatched-sizes",
            pytest.raises(
                ValueError,
                match=r"optics_transmission and opt_trans_wavelengths must have the same length",
            ),
        ),
    ],
)
def test_verify_parameters(
    opt_trans_wavelengths: np.ndarray,
    optics_transmission: np.ndarray | None,
    name: str,
    expectation: AbstractContextManager,
) -> None:
    D = 0  # noqa:N806
    f = 0
    p_x = 0.1
    with expectation:
        sensor = PybsmSensor(
            name,
            D,
            f,
            p_x,
            opt_trans_wavelengths,
            optics_transmission=optics_transmission,
        )

        # testing PybsmSensor call
        assert sensor().D == D
        assert sensor().f == f
        assert sensor().p_x == p_x
        assert sensor().opt_trans_wavelengths.all() == opt_trans_wavelengths.all()
        assert sensor().name == name

        # testing PybsmSensor.create_sensor directily
        assert sensor.create_sensor().D == D
        assert sensor.create_sensor().f == f
        assert sensor.create_sensor().p_x == p_x
        assert sensor.create_sensor().opt_trans_wavelengths.all() == opt_trans_wavelengths.all()
        assert sensor.create_sensor().name == name


def test_config() -> None:
    """Test configuration stability."""
    D = 0  # noqa:N806
    f = 0
    p_x = 0
    opt_trans_wavelengths = np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6
    name = "test"
    inst = PybsmSensor(name, D, f, p_x, opt_trans_wavelengths)
    for i in configuration_test_helper(inst):
        assert i.name == inst.name
        assert i.D == inst.D
        assert i.f == inst.f
        assert i.p_x == inst.p_x
        assert np.array_equal(i.opt_trans_wavelengths, inst.opt_trans_wavelengths)
        assert np.array_equal(i.optics_transmission, inst.optics_transmission)
        assert i.eta == inst.eta
        assert i.w_x == inst.w_x
        assert i.w_y == inst.w_y
        assert i.int_time == inst.int_time
        assert i.n_tdi == inst.n_tdi
        assert i.dark_current == inst.dark_current
        assert i.read_noise == inst.read_noise
        assert i.max_n == inst.max_n
        assert i.bit_depth == inst.bit_depth
        assert i.max_well_fill == inst.max_well_fill
        assert i.s_x == inst.s_x
        assert i.s_y == inst.s_y
        assert i.da_x == inst.da_x
        assert i.da_y == inst.da_y
        assert np.array_equal(i.qe_wavelengths, inst.qe_wavelengths)
        assert np.array_equal(i.qe, inst.qe)


@pytest.fixture()  # noqa:PT001
def sensor_instance() -> PybsmSensor:
    """Fixture to provide a PybsmSensor instance with basic parameters."""
    return PybsmSensor(name="test_sensor", D=0.3, f=1.1, p_x=0.4, opt_trans_wavelengths=np.array([0.1, 0.4]) * 1.0e-6)


# Test cases for the `_check_opt_trans_wavelengths` method
def test_check_opt_trans_wavelengths_valid(sensor_instance: PybsmSensor) -> None:
    """Test that valid opt_trans_wavelengths passes without errors."""
    sensor_instance._check_opt_trans_wavelengths(np.array([0.1, 0.2, 0.4]) * 1.0e-6)


def test_check_opt_trans_wavelengths_too_few_elements(sensor_instance: PybsmSensor) -> None:
    """Test that opt_trans_wavelengths with fewer than two elements raises ValueError."""
    with pytest.raises(ValueError, match="At minimum, at least the start and end wavelengths"):
        sensor_instance._check_opt_trans_wavelengths(np.array([0.1]) * 1.0e-6)


def test_check_opt_trans_wavelengths_not_ascending(sensor_instance: PybsmSensor) -> None:
    """Test that opt_trans_wavelengths not in ascending order raises ValueError."""
    with pytest.raises(ValueError, match="opt_trans_wavelengths must be ascending"):
        sensor_instance._check_opt_trans_wavelengths(np.array([0.4, 0.2, 0.1]) * 1.0e-6)


# Test cases for the `_set_optics_transmission` method
def test_set_optics_transmission_none(sensor_instance: PybsmSensor) -> None:
    """Test that optics_transmission is set to an array of ones if None is provided."""
    sensor_instance._set_optics_transmission()
    assert np.array_equal(sensor_instance.optics_transmission, np.ones(sensor_instance.opt_trans_wavelengths.shape))


def test_set_optics_transmission_valid_array(sensor_instance: PybsmSensor) -> None:
    """Test that a valid optics_transmission array is assigned correctly."""
    valid_transmission = np.array([0.8, 0.9])
    sensor_instance._set_optics_transmission(valid_transmission)
    assert np.array_equal(sensor_instance.optics_transmission, valid_transmission)


def test_set_optics_transmission_mismatched_length(sensor_instance: PybsmSensor) -> None:
    """Test that optics_transmission array with mismatched length raises ValueError."""
    invalid_transmission = np.array([0.8])  # Only one element
    with pytest.raises(ValueError, match="optics_transmission and opt_trans_wavelengths must have the same length"):
        sensor_instance._set_optics_transmission(invalid_transmission)
