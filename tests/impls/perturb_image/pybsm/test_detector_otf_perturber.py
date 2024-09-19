import unittest.mock as mock
from contextlib import nullcontext as does_not_raise
from typing import Any, ContextManager, Dict, Optional

import numpy as np
import pytest
from PIL import Image
from smqtk_core.configuration import configuration_test_helper
from syrupy.assertion import SnapshotAssertion

from nrtk.impls.perturb_image.pybsm.detector_otf_perturber import DetectorOTFPerturber

from ...test_pybsm_utils import (
    TIFFImageSnapshotExtension,
    create_sample_sensor_and_scenario,
)
from ..test_perturber_utils import pybsm_perturber_assertions

INPUT_IMG_FILE = "./examples/pybsm/data/M-41 Walker Bulldog (USA) width 319cm height 272cm.tiff"


@pytest.mark.skipif(
    not DetectorOTFPerturber.is_usable(),
    reason="OpenCV not found. Please install 'nrtk[graphics]' or `nrtk[headless]`.",
)
class TestDetectorOTFPerturber:
    @pytest.mark.parametrize(
        ("use_sensor_scenario", "w_x", "w_y", "f"),
        [(False, None, None, None), (True, 3e-6, 20e-6, 30e-3)],
    )
    def test_reproducibility(
        self,
        use_sensor_scenario: bool,
        w_x: Optional[float],
        w_y: Optional[float],
        f: Optional[float],
    ) -> None:
        """Ensure results are reproducible."""
        img = np.array(Image.open(INPUT_IMG_FILE))
        img_md = {"img_gsd": 3.19 / 160.0}

        sensor = None
        scenario = None
        if use_sensor_scenario:
            sensor, scenario = create_sample_sensor_and_scenario()

        inst = DetectorOTFPerturber(sensor=sensor, scenario=scenario, w_x=w_x, w_y=w_y, f=f)

        out_img = pybsm_perturber_assertions(perturb=inst, image=img, expected=None, additional_params=img_md)

        pybsm_perturber_assertions(perturb=inst, image=img, expected=out_img, additional_params=img_md)

    @pytest.mark.parametrize(
        ("use_sensor_scenario", "additional_params", "expectation"),
        [
            (True, {"img_gsd": 3.19 / 160.0}, does_not_raise()),
            (
                True,
                None,
                pytest.raises(ValueError, match=r"'img_gsd' must be present in image metadata"),
            ),
            (False, {"img_gsd": 3.19 / 160.0}, does_not_raise()),
        ],
    )
    def test_additional_params(
        self,
        use_sensor_scenario: bool,
        additional_params: Dict[str, Any],
        expectation: ContextManager,
    ) -> None:
        """Test that exceptions are appropriately raised based on available metadata."""
        sensor = None
        scenario = None
        if use_sensor_scenario:
            sensor, scenario = create_sample_sensor_and_scenario()
        perturber = DetectorOTFPerturber(sensor=sensor, scenario=scenario)
        img = np.array(Image.open(INPUT_IMG_FILE))
        with expectation:
            _ = perturber.perturb(img, additional_params)

    @pytest.mark.parametrize(
        ("use_sensor_scenario", "w_x", "w_y", "f"),
        [
            (False, None, None, None),
            (True, None, None, None),
            (True, 3e-6, 20e-6, 30e-3),
            (False, 3e-6, 20e-6, 30e-3),
        ],
    )
    def test_configuration(
        self,
        use_sensor_scenario: bool,
        w_x: Optional[float],
        w_y: Optional[float],
        f: Optional[float],
    ) -> None:
        """Test configuration stability."""
        sensor = None
        scenario = None
        if use_sensor_scenario:
            sensor, scenario = create_sample_sensor_and_scenario()

        inst = DetectorOTFPerturber(sensor=sensor, scenario=scenario, w_x=w_x, w_y=w_y, f=f)
        for i in configuration_test_helper(inst):
            if w_x is not None:
                assert i.w_x == w_x
            elif sensor is not None and scenario is not None:
                assert i.w_x == sensor.w_x
            else:  # Default value
                assert i.w_x == 4e-6

            if w_y is not None:
                assert i.w_y == w_y
            elif sensor is not None and scenario is not None:
                assert i.w_y == sensor.w_y
            else:  # Default value
                assert i.w_y == 4e-6

            if f is not None:
                assert i.f == f
            elif sensor is not None and scenario is not None:
                assert i.f == sensor.f
            else:  # Default value
                assert i.f == 50e-3

            if i.sensor is not None and sensor is not None:
                assert i.sensor.name == sensor.name
                assert i.sensor.D == sensor.D
                assert i.sensor.f == sensor.f
                assert i.sensor.p_x == sensor.p_x
                assert np.array_equal(i.sensor.opt_trans_wavelengths, sensor.opt_trans_wavelengths)
                assert np.array_equal(i.sensor.optics_transmission, sensor.optics_transmission)
                assert i.sensor.eta == sensor.eta
                assert i.sensor.w_x == sensor.w_x
                assert i.sensor.w_y == sensor.w_y
                assert i.sensor.int_time == sensor.int_time
                assert i.sensor.dark_current == sensor.dark_current
                assert i.sensor.read_noise == sensor.read_noise
                assert i.sensor.max_n == sensor.max_n
                assert i.sensor.bit_depth == sensor.bit_depth
                assert i.sensor.max_well_fill == sensor.max_well_fill
                assert i.sensor.s_x == sensor.s_x
                assert i.sensor.s_y == sensor.s_y
                assert i.sensor.da_x == sensor.da_x
                assert i.sensor.da_y == sensor.da_y
                assert np.array_equal(i.sensor.qe_wavelengths, sensor.qe_wavelengths)
                assert np.array_equal(i.sensor.qe, sensor.qe)
            else:
                assert i.sensor is None
                assert sensor is None

            if i.scenario is not None and scenario is not None:
                assert i.scenario.name == scenario.name
                assert i.scenario.ihaze == scenario.ihaze
                assert i.scenario.altitude == scenario.altitude
                assert i.scenario.ground_range == scenario.ground_range
                assert i.scenario.aircraft_speed == scenario.aircraft_speed
                assert i.scenario.target_reflectance == scenario.target_reflectance
                assert i.scenario.target_temperature == scenario.target_temperature
                assert i.scenario.background_reflectance == scenario.background_reflectance
                assert i.scenario.background_temperature == scenario.background_temperature
                assert i.scenario.ha_wind_speed == scenario.ha_wind_speed
                assert i.scenario.cn2_at_1m == scenario.cn2_at_1m
            else:
                assert i.scenario is None
                assert scenario is None

    @pytest.mark.parametrize(
        ("use_sensor_scenario", "w_x", "w_y", "f"),
        [
            (False, None, None, None),
            (True, None, None, None),
            (True, 3e-6, 20e-6, 30e-3),
            (False, 3e-6, 20e-6, 30e-3),
        ],
    )
    def test_regression(
        self,
        snapshot: SnapshotAssertion,
        use_sensor_scenario: bool,
        w_x: Optional[float],
        w_y: Optional[float],
        f: Optional[float],
    ) -> None:
        """Regression testing results to detect API changes."""
        img = np.array(Image.open(INPUT_IMG_FILE))
        img_md = {"img_gsd": 3.19 / 160.0}

        sensor = None
        scenario = None
        if use_sensor_scenario:
            sensor, scenario = create_sample_sensor_and_scenario()

        inst = DetectorOTFPerturber(sensor=sensor, scenario=scenario, w_x=w_x, w_y=w_y, f=f)

        out_img = pybsm_perturber_assertions(perturb=inst, image=img, expected=None, additional_params=img_md)

        assert TIFFImageSnapshotExtension.ndarray2bytes(out_img) == snapshot(extension_class=TIFFImageSnapshotExtension)


@mock.patch.object(DetectorOTFPerturber, "is_usable")
def test_missing_deps(mock_is_usable) -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    mock_is_usable.return_value = False
    assert not DetectorOTFPerturber.is_usable()
    with pytest.raises(ImportError, match=r"OpenCV not found"):
        DetectorOTFPerturber()
