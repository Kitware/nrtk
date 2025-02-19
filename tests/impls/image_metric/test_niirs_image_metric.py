import math
import unittest.mock as mock

import pytest

from nrtk.impls.image_metric.niirs_image_metric import NIIRSImageMetric
from nrtk.utils._exceptions import PyBSMAndOpenCVImportError
from tests.impls.test_pybsm_utils import create_sample_sensor_and_scenario


@pytest.mark.skipif(
    not NIIRSImageMetric.is_usable(),
    reason="not NIIRSImageMetric.is_usable()",
)
class TestSNRImageMetric:
    """This class contains the unit tests for the functionality of the NIIRSImageMetric impl."""

    def test_consistency(self) -> None:
        expected_niirs = 5.64306090838768
        sensor, scenario = create_sample_sensor_and_scenario()
        niirs_metric = NIIRSImageMetric(sensor=sensor, scenario=scenario)
        # Test metric interface directly
        niirs = niirs_metric.compute()
        assert math.isclose(expected_niirs, niirs)

        # Test callable
        assert math.isclose(expected_niirs, niirs_metric())

    def test_reproducibility(self) -> None:
        """Ensure results are reproducible."""
        # Test metric interface directly
        sensor, scenario = create_sample_sensor_and_scenario()

        niirs_ouput = NIIRSImageMetric(sensor=sensor, scenario=scenario).compute()
        niirs_metric = NIIRSImageMetric(sensor=sensor, scenario=scenario)

        assert niirs_ouput == niirs_metric.compute()

    def test_get_config(self) -> None:
        """Ensure get_config is correct."""
        sensor, scenario = create_sample_sensor_and_scenario()
        niirs_metric = NIIRSImageMetric(sensor=sensor, scenario=scenario)

        niirs_metric_config = niirs_metric.get_config()
        assert niirs_metric_config["sensor"] == {
            "nrtk.impls.perturb_image.pybsm.sensor.PybsmSensor": sensor.get_config(),
            "type": "nrtk.impls.perturb_image.pybsm.sensor.PybsmSensor",
        }
        assert niirs_metric_config["scenario"] == {
            "nrtk.impls.perturb_image.pybsm.scenario.PybsmScenario": scenario.get_config(),
            "type": "nrtk.impls.perturb_image.pybsm.scenario.PybsmScenario",
        }

    def test_classname(self) -> None:
        sensor, scenario = create_sample_sensor_and_scenario()
        niirs_metric = NIIRSImageMetric(sensor=sensor, scenario=scenario)
        assert niirs_metric.name == "NIIRSImageMetric"

    @mock.patch.object(NIIRSImageMetric, "is_usable")
    def test_missing_deps(self, mock_is_usable: mock.MagicMock) -> None:
        """Test that an exception is raised when required dependencies are not installed."""
        mock_is_usable.return_value = False
        assert not NIIRSImageMetric.is_usable()
        sensor, scenario = create_sample_sensor_and_scenario()
        with pytest.raises(PyBSMAndOpenCVImportError):
            NIIRSImageMetric(sensor=sensor, scenario=scenario)
