from __future__ import annotations

import unittest.mock as mock
from collections.abc import Hashable, Iterable, Sequence
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image
from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from syrupy.assertion import SnapshotAssertion

from nrtk.impls.perturb.optical.circular_aperture_otf_perturber import (
    CircularApertureOTFPerturber,
)
from nrtk.utils._exceptions import PyBSMImportError
from tests.impls import INPUT_TANK_IMG_FILE_PATH as INPUT_IMG_FILE_PATH
from tests.impls.perturb.test_perturber_utils import pybsm_perturber_assertions
from tests.impls.test_pybsm_utils import create_sample_sensor_and_scenario


@pytest.mark.skipif(not CircularApertureOTFPerturber.is_usable(), reason=str(PyBSMImportError()))
class TestCircularApertureOTFPerturber:
    def test_interp_consistency(self) -> None:
        """Run on a dummy image to ensure output matches precomputed results."""
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        img_gsd = 3.19 / 160.0
        sensor, scenario = create_sample_sensor_and_scenario()
        # Test perturb interface directly
        inst = CircularApertureOTFPerturber(sensor=sensor, scenario=scenario, interp=True)
        inst2 = CircularApertureOTFPerturber(sensor=sensor, scenario=scenario, interp=False)
        out_image = pybsm_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=None,
            img_gsd=img_gsd,
        )

        pybsm_perturber_assertions(
            perturb=inst2.perturb,
            image=image,
            expected=out_image,
            img_gsd=img_gsd,
        )

    @pytest.mark.parametrize(
        ("mtf_wavelengths", "mtf_weights", "D", "eta", "interp"),
        [
            ([0.5e-6, 0.6e-6], [0.5, 0.5], None, None, True),
            ([0.5e-6, 0.6e-6], [0.5, 0.5], 0.275, 0.4, True),
            ([0.2e-6, 0.4e-6, 0.6e-6], [1.0, 0.5, 1.0], 0.4, 0.1, False),
        ],
    )
    def test_provided_reproducibility(
        self,
        mtf_wavelengths: Sequence[float],
        mtf_weights: Sequence[float],
        D: float,  # noqa N802
        eta: float,
        interp: bool,
    ) -> None:
        """Ensure results are reproducible."""
        # Test perturb interface directly
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        sensor, scenario = create_sample_sensor_and_scenario()
        inst = CircularApertureOTFPerturber(
            sensor=sensor,
            scenario=scenario,
            mtf_wavelengths=mtf_wavelengths,
            mtf_weights=mtf_weights,
            D=D,
            eta=eta,
            interp=interp,
        )
        img_gsd = 3.19 / 160.0
        out_image = pybsm_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=None,
            img_gsd=img_gsd,
        )
        pybsm_perturber_assertions(
            perturb=inst.perturb,
            image=image,
            expected=out_image,
            img_gsd=img_gsd,
        )

    def test_default_reproducibility(self) -> None:
        """Ensure results are reproducible."""
        # Test perturb interface directly
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        inst = CircularApertureOTFPerturber()
        out_image = pybsm_perturber_assertions(perturb=inst.perturb, image=image, expected=None)
        pybsm_perturber_assertions(perturb=inst.perturb, image=image, expected=out_image)

    @pytest.mark.parametrize(
        ("additional_params", "expectation"),
        [
            ({"img_gsd": 3.19 / 160.0}, does_not_raise()),
            (
                {},
                pytest.raises(ValueError, match=r"'img_gsd' must be provided"),
            ),
        ],
    )
    def test_provided_additional_params(
        self,
        additional_params: dict[str, Any],
        expectation: AbstractContextManager,
    ) -> None:
        """Test variations of additional params."""
        sensor, scenario = create_sample_sensor_and_scenario()
        perturber = CircularApertureOTFPerturber(sensor=sensor, scenario=scenario)
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        with expectation:
            _ = perturber(image, **additional_params)

    @pytest.mark.parametrize(
        ("mtf_wavelengths", "mtf_weights", "D", "eta", "interp", "expectation"),
        [
            ([0.5e-6, 0.6e-6], [0.5, 0.5], 0.003, 0.0, True, does_not_raise()),
            (
                [0.5e-6, 0.6e-6],
                [],
                0.003,
                0.0,
                False,
                pytest.raises(ValueError, match=r"mtf_weights is empty"),
            ),
            (
                [],
                [0.5, 0.5],
                0.003,
                0.0,
                True,
                pytest.raises(ValueError, match=r"mtf_wavelengths is empty"),
            ),
            (
                [0.5e-6, 0.6e-6],
                [0.5],
                0.003,
                0.0,
                True,
                pytest.raises(
                    ValueError,
                    match=r"mtf_wavelengths and mtf_weights are not the same length",
                ),
            ),
        ],
    )
    def test_configuration_bounds(
        self,
        mtf_wavelengths: Sequence[float],
        mtf_weights: Sequence[float],
        D: float,  # noqa N802
        eta: float,
        interp: bool,
        expectation: AbstractContextManager,
    ) -> None:
        """Test variations of additional params."""
        with expectation:
            _ = CircularApertureOTFPerturber(
                mtf_wavelengths=mtf_wavelengths,
                mtf_weights=mtf_weights,
                D=D,
                eta=eta,
                interp=interp,
            )

    @pytest.mark.parametrize(
        ("additional_params", "expectation"),
        [
            ({}, does_not_raise()),
        ],
    )
    def test_default_additional_params(
        self,
        additional_params: dict[str, Any],
        expectation: AbstractContextManager,
    ) -> None:
        """Test variations of additional params."""
        perturber = CircularApertureOTFPerturber()
        image = np.array(Image.open(INPUT_IMG_FILE_PATH))
        with expectation:
            _ = perturber(image, **additional_params)

    @pytest.mark.parametrize(
        ("mtf_wavelengths", "mtf_weights", "D", "eta"),
        [
            ([0.5e-6, 0.6e-6], [0.5, 0.5], 0.003, 0.0),
        ],
    )
    def test_parameter_configuration(
        self,
        mtf_wavelengths: Sequence[float],
        mtf_weights: Sequence[float],
        D: float,  # noqa N802
        eta: float,
    ) -> None:
        """Test configuration stability."""
        inst = CircularApertureOTFPerturber(mtf_wavelengths=mtf_wavelengths, mtf_weights=mtf_weights, D=D, eta=eta)
        for i in configuration_test_helper(inst):
            assert i.mtf_wavelengths is not None
            assert i.mtf_weights is not None
            assert np.array_equal(i.mtf_wavelengths, mtf_wavelengths)
            assert np.array_equal(i.mtf_weights, mtf_weights)
            assert i.D == D
            assert i.eta == eta

    def test_sensor_scenario_configuration(self) -> None:
        """Test configuration stability."""
        sensor, scenario = create_sample_sensor_and_scenario()
        inst = CircularApertureOTFPerturber(sensor=sensor, scenario=scenario)
        for i in configuration_test_helper(inst):
            if i.sensor:
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
            if i.scenario:
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

    @pytest.mark.parametrize(
        ("mtf_wavelengths", "mtf_weights", "D", "eta"),
        [
            ([0.5e-6, 0.6e-6], [0.5, 0.5], 0.003, 0.0),
        ],
    )
    def test_overall_configuration(
        self,
        mtf_wavelengths: Sequence[float],
        mtf_weights: Sequence[float],
        D: float,  # noqa N802
        eta: float,
    ) -> None:
        """Test configuration stability."""
        sensor, scenario = create_sample_sensor_and_scenario()
        inst = CircularApertureOTFPerturber(
            sensor=sensor,
            scenario=scenario,
            mtf_wavelengths=mtf_wavelengths,
            mtf_weights=mtf_weights,
            D=D,
            eta=eta,
        )
        for i in configuration_test_helper(inst):
            assert i.mtf_wavelengths is not None
            assert i.mtf_weights is not None
            assert np.array_equal(i.mtf_wavelengths, mtf_wavelengths)
            assert np.array_equal(i.mtf_weights, mtf_weights)
            assert i.D == D
            assert i.eta == eta
            if i.sensor:
                assert i.sensor.name == sensor.name
                assert i.sensor.D == D
                assert i.sensor.f == sensor.f
                assert i.sensor.p_x == sensor.p_x
                assert np.array_equal(i.sensor.opt_trans_wavelengths, sensor.opt_trans_wavelengths)
                assert np.array_equal(i.sensor.optics_transmission, sensor.optics_transmission)
                assert i.sensor.eta == eta
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
            if i.scenario:
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

    @pytest.mark.parametrize(
        ("use_sensor_scenario", "mtf_wavelengths", "mtf_weights", "D", "eta", "interp", "is_rgb"),
        [
            (False, None, None, None, None, None, False),
            (True, None, None, None, None, None, True),
            (False, None, None, None, None, None, True),
            (True, None, None, None, None, None, False),
            (True, [0.5e-6, 0.6e-6], [0.5, 0.5], 0.275, 0.4, False, True),
            (False, [0.5e-6, 0.6e-6], [0.5, 0.5], 0.003, 0.0, True, False),
            (True, [0.5e-6, 0.6e-6], [0.5, 0.5], 0.275, 0.4, False, False),
            (False, [0.5e-6, 0.6e-6], [0.5, 0.5], 0.003, 0.0, True, True),
        ],
    )
    def test_regression(
        self,
        psnr_tiff_snapshot: SnapshotAssertion,
        use_sensor_scenario: bool,
        mtf_wavelengths: Sequence[float] | None,
        mtf_weights: Sequence[float] | None,
        D: float | None,  # noqa N802
        eta: float | None,
        interp: bool,
        is_rgb: bool,
    ) -> None:
        """Regression testing results to detect API changes."""
        img = np.array(Image.open(INPUT_IMG_FILE_PATH))
        if is_rgb:
            img = np.stack((img,) * 3, axis=-1)
        img_md = {"img_gsd": 3.19 / 160.0}

        sensor = None
        scenario = None
        if use_sensor_scenario:
            sensor, scenario = create_sample_sensor_and_scenario()

        inst = CircularApertureOTFPerturber(
            sensor=sensor,
            scenario=scenario,
            mtf_wavelengths=mtf_wavelengths,
            mtf_weights=mtf_weights,
            D=D,
            eta=eta,
            interp=interp,
        )

        out_img = pybsm_perturber_assertions(perturb=inst, image=img, expected=None, **img_md)

        psnr_tiff_snapshot.assert_match(out_img)

    @pytest.mark.parametrize(
        "boxes",
        [
            None,
            [(AxisAlignedBoundingBox((0, 0), (1, 1)), {"test": 0.0})],
            [
                (AxisAlignedBoundingBox((0, 0), (1, 1)), {"test": 0.0}),
                (AxisAlignedBoundingBox((2, 2), (3, 3)), {"test2": 1.0}),
            ],
        ],
    )
    def test_perturb_with_boxes(self, boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]) -> None:
        """Test that bounding boxes do not change during perturb."""
        inst = CircularApertureOTFPerturber()
        _, out_boxes = inst.perturb(np.ones((256, 256, 3)), boxes=boxes, img_gsd=(3.19 / 160))
        assert boxes == out_boxes

    @pytest.mark.parametrize(
        ("param_name", "param_value"),
        [
            ("ground_range", 10000),
            ("altitude", 5000),
            ("ihaze", 2),
        ],
    )
    def test_kwargs_application(self, param_name: str, param_value: int) -> None:
        """Test that kwargs are properly applied to sensor/scenario."""
        sensor, scenario = create_sample_sensor_and_scenario()
        inst = CircularApertureOTFPerturber(sensor=sensor, scenario=scenario, **{param_name: param_value})  # type: ignore[arg-type]

        # Verify the parameter was applied
        assert hasattr(inst.sensor, param_name) or hasattr(inst.scenario, param_name)
        if hasattr(inst.scenario, param_name):
            assert getattr(inst.scenario, param_name) == param_value
        elif hasattr(inst.sensor, param_name):
            assert getattr(inst.sensor, param_name) == param_value

        # Verify params property returns the kwargs
        assert inst.params == {param_name: param_value}


@mock.patch.object(CircularApertureOTFPerturber, "is_usable")
def test_missing_deps(mock_is_usable: MagicMock) -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    mock_is_usable.return_value = False
    assert not CircularApertureOTFPerturber.is_usable()
    with pytest.raises(PyBSMImportError):
        CircularApertureOTFPerturber()
