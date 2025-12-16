import math
import unittest.mock as mock

import numpy as np
import pytest

from nrtk.impls.image_metric.niirs_image_metric import NIIRSImageMetric
from nrtk.utils._exceptions import PyBSMImportError
from nrtk.utils._import_guard import import_guard
from tests.impls.test_pybsm_utils import create_sample_sensor_and_scenario

is_usable: bool = import_guard(module_name="pybsm", exception=PyBSMImportError, submodules=["otf"])


@pytest.mark.skipif(
    not NIIRSImageMetric.is_usable(),
    reason="not NIIRSImageMetric.is_usable()",
)
class TestSNRImageMetric:
    """This class contains the unit tests for the functionality of the NIIRSImageMetric impl."""

    def test_consistency(self) -> None:
        expected_niirs = 5.619442360319594
        sensor_and_scenario = create_sample_sensor_and_scenario()
        niirs_metric = NIIRSImageMetric(**sensor_and_scenario)
        # Test metric interface directly
        niirs = niirs_metric.compute()
        assert math.isclose(expected_niirs, niirs)

        # Test callable
        assert math.isclose(expected_niirs, niirs_metric())

    def test_reproducibility(self) -> None:
        """Ensure results are reproducible."""
        # Test metric interface directly
        sensor_and_scenario = create_sample_sensor_and_scenario()

        niirs_ouput = NIIRSImageMetric(**sensor_and_scenario).compute()
        niirs_metric = NIIRSImageMetric(**sensor_and_scenario)

        assert niirs_ouput == niirs_metric.compute()

    def test_get_config(self) -> None:
        """Ensure get_config is correct."""
        sensor_and_scenario = create_sample_sensor_and_scenario()
        niirs_metric = NIIRSImageMetric(**sensor_and_scenario)

        niirs_metric_config = niirs_metric.get_config()
        assert niirs_metric_config["sensor_name"] == sensor_and_scenario["sensor_name"]
        assert niirs_metric_config["D"] == sensor_and_scenario["D"]
        assert niirs_metric_config["f"] == sensor_and_scenario["f"]
        assert niirs_metric_config["p_x"] == sensor_and_scenario["p_x"]
        assert np.array_equal(
            niirs_metric_config["opt_trans_wavelengths"],
            sensor_and_scenario["opt_trans_wavelengths"],
        )
        assert niirs_metric_config["scenario_name"] == sensor_and_scenario["scenario_name"]
        assert niirs_metric_config["ihaze"] == sensor_and_scenario["ihaze"]
        assert niirs_metric_config["altitude"] == sensor_and_scenario["altitude"]
        assert niirs_metric_config["ground_range"] == sensor_and_scenario["ground_range"]
        assert np.array_equal(niirs_metric_config["optics_transmission"], sensor_and_scenario["optics_transmission"])
        assert niirs_metric_config["eta"] == sensor_and_scenario["eta"]
        assert niirs_metric_config["w_x"] == sensor_and_scenario["w_x"]
        assert niirs_metric_config["w_y"] == sensor_and_scenario["w_y"]
        assert niirs_metric_config["int_time"] == sensor_and_scenario["int_time"]
        assert niirs_metric_config["n_tdi"] == sensor_and_scenario["n_tdi"]
        assert niirs_metric_config["dark_current"] == sensor_and_scenario["dark_current"]
        assert niirs_metric_config["read_noise"] == sensor_and_scenario["read_noise"]
        assert niirs_metric_config["max_n"] == sensor_and_scenario["max_n"]
        assert niirs_metric_config["bit_depth"] == sensor_and_scenario["bit_depth"]
        assert niirs_metric_config["max_well_fill"] == sensor_and_scenario["max_well_fill"]
        assert niirs_metric_config["s_x"] == sensor_and_scenario["s_x"]
        assert niirs_metric_config["s_y"] == sensor_and_scenario["s_y"]
        assert np.array_equal(niirs_metric_config["qe_wavelengths"], sensor_and_scenario["qe_wavelengths"])
        assert np.array_equal(niirs_metric_config["qe"], sensor_and_scenario["qe"])
        assert niirs_metric_config["aircraft_speed"] == sensor_and_scenario["aircraft_speed"]

    def test_classname(self) -> None:
        sensor_and_scenario = create_sample_sensor_and_scenario()
        niirs_metric = NIIRSImageMetric(**sensor_and_scenario)
        assert niirs_metric.name == "NIIRSImageMetric"

    @mock.patch.object(NIIRSImageMetric, "is_usable")
    def test_missing_deps(self, mock_is_usable: mock.MagicMock) -> None:
        """Test that an exception is raised when required dependencies are not installed."""
        mock_is_usable.return_value = False
        assert not NIIRSImageMetric.is_usable()
        sensor_and_scenario = create_sample_sensor_and_scenario()
        with pytest.raises(PyBSMImportError):
            NIIRSImageMetric(**sensor_and_scenario)
