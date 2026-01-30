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

from nrtk.impls.perturb_image.optical.turbulence_aperture_otf_perturber import (
    TurbulenceApertureOTFPerturber,
)
from nrtk.utils._exceptions import PyBSMImportError
from tests.impls import INPUT_TANK_IMG_FILE_PATH as INPUT_IMG_FILE_PATH
from tests.impls.perturb_image.test_perturber_utils import pybsm_perturber_assertions
from tests.utils.test_pybsm import create_sample_sensor, create_sample_sensor_and_scenario


@pytest.mark.skipif(not TurbulenceApertureOTFPerturber.is_usable(), reason=str(PyBSMImportError()))
class TestTurbulenceApertureOTFPerturber:
    @pytest.mark.parametrize(
        (
            "use_sensor_scenario",
            "mtf_wavelengths",
            "mtf_weights",
            "altitude",
            "slant_range",
            "D",
            "ha_wind_speed",
            "cn2_at_1m",
            "int_time",
            "n_tdi",
            "aircraft_speed",
            "interp",
        ),
        [
            (False, None, None, None, None, None, None, None, None, None, None, None),
            (
                True,
                [0.50e-6, 0.66e-6],
                [1.0, 1.0],
                250,
                250,
                40e-3,
                0,
                1.7e-14,
                30e-3,
                1.0,
                0,
                True,
            ),
            (
                True,
                [0.50e-6, 0.66e-6],
                [1.0, 1.0],
                250,
                250,
                40e-3,
                0,
                1.7e-14,
                30e-3,
                1.0,
                0,
                False,
            ),
        ],
    )
    def test_reproducibility(  # noqa: C901
        self,
        use_sensor_scenario: bool,
        mtf_wavelengths: Sequence[float] | None,
        mtf_weights: Sequence[float] | None,
        altitude: float | None,
        slant_range: float | None,
        D: float | None,  # noqa: N803
        ha_wind_speed: float | None,
        cn2_at_1m: float | None,
        int_time: float | None,
        n_tdi: float | None,
        aircraft_speed: float | None,
        interp: bool,
    ) -> None:
        """Ensure results are reproducible."""
        img = np.array(Image.open(INPUT_IMG_FILE_PATH))
        img_md = {"img_gsd": 3.19 / 160.0}

        sensor_and_scenario = {}
        if use_sensor_scenario:
            sensor_and_scenario = create_sample_sensor_and_scenario()

        if altitude is not None:
            sensor_and_scenario["altitude"] = altitude
        if slant_range is not None:
            sensor_and_scenario["slant_range"] = slant_range
        if D is not None:
            sensor_and_scenario["D"] = D
        if ha_wind_speed is not None:
            sensor_and_scenario["ha_wind_speed"] = ha_wind_speed
        if cn2_at_1m is not None:
            sensor_and_scenario["cn2_at_1m"] = cn2_at_1m
        if int_time is not None:
            sensor_and_scenario["int_time"] = int_time
        if n_tdi is not None:
            sensor_and_scenario["n_tdi"] = n_tdi
        if aircraft_speed is not None:
            sensor_and_scenario["aircraft_speed"] = aircraft_speed

        inst = TurbulenceApertureOTFPerturber(
            mtf_wavelengths=mtf_wavelengths,
            mtf_weights=mtf_weights,
            interp=interp,
            **sensor_and_scenario,
        )

        out_img = pybsm_perturber_assertions(perturb=inst, image=img, expected=None, **img_md)

        pybsm_perturber_assertions(perturb=inst, image=img, expected=out_img, **img_md)

    @pytest.mark.parametrize(
        ("use_sensor_scenario", "kwargs", "expectation"),
        [
            (True, {"img_gsd": 3.19 / 160.0}, does_not_raise()),
            (
                True,
                dict(),
                pytest.raises(ValueError, match=r"'img_gsd' must be provided"),
            ),
            (False, {"img_gsd": 3.19 / 160.0}, does_not_raise()),
        ],
    )
    def test_kwargs(
        self,
        use_sensor_scenario: bool,
        kwargs: dict[str, Any],
        expectation: AbstractContextManager,
    ) -> None:
        """Test that exceptions are appropriately raised based on available metadata."""
        sensor_and_scenario = {}
        if use_sensor_scenario:
            sensor_and_scenario = create_sample_sensor_and_scenario()
        perturber = TurbulenceApertureOTFPerturber(**sensor_and_scenario)
        img = np.array(Image.open(INPUT_IMG_FILE_PATH))
        with expectation:
            _ = perturber.perturb(image=img, **kwargs)

    @pytest.mark.parametrize(
        ("mtf_wavelengths", "mtf_weights", "cn2_at_1m", "expectation"),
        [
            ([0.5e-6, 0.6e-6], [0.5, 0.5], 0.1, does_not_raise()),
            (
                [0.5e-6, 0.6e-6],
                list(),
                0.1,
                pytest.raises(ValueError, match=r"mtf_weights is empty"),
            ),
            (
                list(),
                [0.5, 0.5],
                0.1,
                pytest.raises(ValueError, match=r"mtf_wavelengths is empty"),
            ),
            (
                [0.5e-6, 0.6e-6],
                [0.5],
                0.1,
                pytest.raises(
                    ValueError,
                    match=r"mtf_wavelengths and mtf_weights are not the same length",
                ),
            ),
            (
                [0.5e-6, 0.6e-6],
                [0.5, 0.5],
                0,
                pytest.raises(
                    ValueError,
                    match=r"Turbulence effect cannot be applied at ground level",
                ),
            ),
        ],
    )
    def test_configuration_bounds(
        self,
        mtf_wavelengths: Sequence[float],
        mtf_weights: Sequence[float],
        cn2_at_1m: float,
        expectation: AbstractContextManager,
    ) -> None:
        """Raise appropriate errors for specific parameters."""
        with expectation:
            _ = TurbulenceApertureOTFPerturber(
                mtf_wavelengths=mtf_wavelengths,
                mtf_weights=mtf_weights,
                cn2_at_1m=cn2_at_1m,
            )

    @pytest.mark.parametrize(
        (
            "use_sensor_scenario",
            "mtf_wavelengths",
            "mtf_weights",
            "altitude",
            "slant_range",
            "D",
            "ha_wind_speed",
            "cn2_at_1m",
            "int_time",
            "n_tdi",
            "aircraft_speed",
            "interp",
        ),
        [
            (False, None, None, None, None, None, None, None, None, None, None, None),
            (True, None, None, None, None, None, None, None, None, None, None, None),
            (True, [0.50e-6, 0.66e-6], [1.0, 1.0], 250, 250, 40e-3, 0, 1.7e-14, 30e-3, 1.0, 0, True),
            (True, [0.50e-6, 0.66e-6], [1.0, 1.0], 250, 250, 40e-3, 0, 1.7e-14, 30e-3, 1.0, 0, False),
            (False, [0.50e-6, 0.66e-6], [1.0, 1.0], 250, 250, 40e-3, 0, 1.7e-14, 30e-3, 1.0, 0, False),
        ],
    )
    def test_configuration(  # noqa: C901
        self,
        use_sensor_scenario: bool,
        mtf_wavelengths: Sequence[float] | None,
        mtf_weights: Sequence[float] | None,
        altitude: float | None,
        slant_range: float | None,
        D: float | None,  # noqa: N803
        ha_wind_speed: float | None,
        cn2_at_1m: float | None,
        int_time: float | None,
        n_tdi: float | None,
        aircraft_speed: float | None,
        interp: bool,
    ) -> None:
        """Test configuration stability."""
        import pybsm.radiance as radiance
        from pybsm.simulation.sensor import Sensor
        from pybsm.utils import load_database_atmosphere

        sensor_and_scenario = {}
        wavelengths = np.asarray(list())
        weights = np.asarray(list())
        pos_weights = np.asarray(list())
        if use_sensor_scenario:
            sensor_and_scenario = create_sample_sensor_and_scenario()
            # Multiple type ignores added for pyright's handling of guarded imports
            atm = load_database_atmosphere(
                altitude=sensor_and_scenario["altitude"],
                ground_range=sensor_and_scenario["ground_range"],
                ihaze=sensor_and_scenario["ihaze"],
            )
            sensor_params = create_sample_sensor()

            sensor = Sensor(
                name=sensor_params["sensor_name"],
                D=sensor_params["D"],
                f=sensor_params["f"],
                p_x=sensor_params["p_x"],
                opt_trans_wavelengths=sensor_params["opt_trans_wavelengths"],
            )

            sensor.optics_transmission = sensor_params["optics_transmission"]
            sensor.eta = sensor_params["eta"]
            sensor.p_y = sensor_params["p_x"]
            sensor.w_x = sensor_params["w_x"]
            sensor.w_y = sensor_params["w_y"]
            sensor.s_x = sensor_params["s_x"]
            sensor.s_y = sensor_params["s_y"]
            sensor.int_time = sensor_params["int_time"]
            sensor.n_tdi = sensor_params["n_tdi"]
            sensor.dark_current = sensor_params["dark_current"]
            sensor.read_noise = sensor_params["read_noise"]
            sensor.max_n = sensor_params["max_n"]
            sensor.max_well_fill = sensor_params["max_well_fill"]
            sensor.bit_depth = sensor_params["bit_depth"]
            sensor.qe_wavelengths = sensor_params["qe_wavelengths"]
            sensor.qe = sensor_params["qe"]
            _, _, spectral_weights = radiance.reflectance_to_photoelectrons(
                atm=atm,
                sensor=sensor,
                int_time=sensor_and_scenario["int_time"],
            )

            wavelengths = spectral_weights[0]
            weights = spectral_weights[1]

            # cut down the wavelength range to only the regions of interests
            pos_weights = np.where(weights > 0.0)

        if altitude is not None:
            sensor_and_scenario["altitude"] = altitude
        if slant_range is not None:
            sensor_and_scenario["slant_range"] = slant_range
        if D is not None:
            sensor_and_scenario["D"] = D
        if ha_wind_speed is not None:
            sensor_and_scenario["ha_wind_speed"] = ha_wind_speed
        if cn2_at_1m is not None:
            sensor_and_scenario["cn2_at_1m"] = cn2_at_1m
        if int_time is not None:
            sensor_and_scenario["int_time"] = int_time
        if n_tdi is not None:
            sensor_and_scenario["n_tdi"] = n_tdi
        if aircraft_speed is not None:
            sensor_and_scenario["aircraft_speed"] = aircraft_speed

        inst = TurbulenceApertureOTFPerturber(
            mtf_wavelengths=mtf_wavelengths,
            mtf_weights=mtf_weights,
            interp=interp,
            **sensor_and_scenario,
        )

        for i in configuration_test_helper(inst):
            # MTF wavelengths and weights
            if mtf_wavelengths is not None:
                assert np.array_equal(i.mtf_wavelengths, mtf_wavelengths)
            elif use_sensor_scenario:
                assert np.array_equal(i.mtf_wavelengths, wavelengths[pos_weights])
            else:  # Default value
                assert np.allclose(i.mtf_wavelengths, np.array([0.50e-6, 0.66e-6]))
            if mtf_weights is not None:
                assert np.array_equal(i.mtf_weights, mtf_weights)
            elif use_sensor_scenario:
                assert np.array_equal(i.mtf_weights, weights[pos_weights])
            else:  # Default value
                assert np.allclose(i.mtf_weights, np.array([1.0, 1.0]))

            # Sensor parameters
            if D is not None:
                assert i.D == D
            elif use_sensor_scenario:
                assert sensor_and_scenario["D"] == i.D
            else:  # Default value
                assert i.D == 40e-3
            if int_time is not None:
                assert i.int_time == int_time
            elif use_sensor_scenario:
                assert i.int_time == sensor_and_scenario["int_time"]
            else:  # Default value
                assert i.int_time == 30e-3
            if n_tdi is not None:
                assert i.n_tdi == n_tdi
            elif use_sensor_scenario:
                assert i.n_tdi == sensor_and_scenario["n_tdi"]
            else:  # Default value
                assert i.n_tdi == 1.0

            # Scenario parameters
            if altitude is not None:
                assert i.altitude == altitude
            elif use_sensor_scenario:
                assert i.altitude == sensor_and_scenario["altitude"]
            else:  # Default value
                assert i.altitude == 250
            if ha_wind_speed is not None:
                assert i.ha_wind_speed == ha_wind_speed
            elif use_sensor_scenario:
                assert i.ha_wind_speed == 21.0
            else:  # Default value
                assert i.ha_wind_speed == 0
            if cn2_at_1m is not None:
                assert i.cn2_at_1m == cn2_at_1m
            else:  # Default value
                assert i.cn2_at_1m == 1.7e-14
            if aircraft_speed is not None:
                assert i.aircraft_speed == aircraft_speed
            elif use_sensor_scenario:
                assert i.aircraft_speed == sensor_and_scenario["aircraft_speed"]
            else:  # Default value
                assert i.aircraft_speed == 0

    @pytest.mark.parametrize(
        (
            "use_sensor_scenario",
            "mtf_wavelengths",
            "mtf_weights",
            "altitude",
            "slant_range",
            "D",
            "ha_wind_speed",
            "cn2_at_1m",
            "int_time",
            "n_tdi",
            "aircraft_speed",
            "interp",
            "is_rgb",
        ),
        [
            (False, None, None, None, None, None, None, None, None, None, None, None, True),
            (True, None, None, None, None, None, None, None, None, None, None, None, False),
            (False, None, None, None, None, None, None, None, None, None, None, None, False),
            (True, None, None, None, None, None, None, None, None, None, None, None, True),
            (True, [0.50e-6, 0.66e-6], [1.0, 1.0], 250, 250, 40e-3, 0, 1.7e-14, 30e-3, 1.0, 0, True, True),
            (False, [0.50e-6, 0.66e-6], [1.0, 1.0], 250, 250, 40e-3, 0, 1.7e-14, 30e-3, 1.0, 0, False, False),
            (True, [0.50e-6, 0.66e-6], [1.0, 1.0], 250, 250, 40e-3, 0, 1.7e-14, 30e-3, 1.0, 0, True, False),
            (False, [0.50e-6, 0.66e-6], [1.0, 1.0], 250, 250, 40e-3, 0, 1.7e-14, 30e-3, 1.0, 0, False, True),
        ],
    )
    def test_regression(  # noqa: C901
        self,
        psnr_tiff_snapshot: SnapshotAssertion,
        use_sensor_scenario: bool,
        mtf_wavelengths: Sequence[float] | None,
        mtf_weights: Sequence[float] | None,
        altitude: float | None,
        slant_range: float | None,
        D: float | None,  # noqa: N803
        ha_wind_speed: float | None,
        cn2_at_1m: float | None,
        int_time: float | None,
        n_tdi: float | None,
        aircraft_speed: float | None,
        interp: bool,
        is_rgb: bool,
    ) -> None:
        """Regression testing results to detect API changes."""
        img = np.array(Image.open(INPUT_IMG_FILE_PATH))
        if is_rgb:
            img = np.stack((img,) * 3, axis=-1)
        img_md = {"img_gsd": 3.19 / 160.0}

        sensor_and_scenario = {}
        if use_sensor_scenario:
            sensor_and_scenario = create_sample_sensor_and_scenario()

        if altitude is not None:
            sensor_and_scenario["altitude"] = altitude
        if slant_range is not None:
            sensor_and_scenario["slant_range"] = slant_range
        if D is not None:
            sensor_and_scenario["D"] = D
        if ha_wind_speed is not None:
            sensor_and_scenario["ha_wind_speed"] = ha_wind_speed
        if cn2_at_1m is not None:
            sensor_and_scenario["cn2_at_1m"] = cn2_at_1m
        if int_time is not None:
            sensor_and_scenario["int_time"] = int_time
        if n_tdi is not None:
            sensor_and_scenario["n_tdi"] = n_tdi
        if aircraft_speed is not None:
            sensor_and_scenario["aircraft_speed"] = aircraft_speed

        inst = TurbulenceApertureOTFPerturber(
            mtf_wavelengths=mtf_wavelengths,
            mtf_weights=mtf_weights,
            interp=interp,
            **sensor_and_scenario,
        )

        out_img = pybsm_perturber_assertions(perturb=inst, image=img, expected=None, **img_md)
        psnr_tiff_snapshot.assert_match(out_img)

    @pytest.mark.parametrize(
        "boxes",
        [
            None,
            [(AxisAlignedBoundingBox(min_vertex=(0, 0), max_vertex=(1, 1)), {"test": 0.0})],
            [
                (AxisAlignedBoundingBox(min_vertex=(0, 0), max_vertex=(1, 1)), {"test": 0.0}),
                (AxisAlignedBoundingBox(min_vertex=(2, 2), max_vertex=(3, 3)), {"test2": 1.0}),
            ],
        ],
    )
    def test_perturb_with_boxes(self, boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]) -> None:
        """Test that bounding boxes do not change during perturb."""
        inst = TurbulenceApertureOTFPerturber()
        _, out_boxes = inst.perturb(image=np.ones((256, 256, 3)), boxes=boxes, img_gsd=(3.19 / 160))
        assert boxes == out_boxes


@mock.patch.object(TurbulenceApertureOTFPerturber, "is_usable")
def test_missing_deps(mock_is_usable: MagicMock) -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    mock_is_usable.return_value = False
    assert not TurbulenceApertureOTFPerturber.is_usable()
    with pytest.raises(PyBSMImportError):
        TurbulenceApertureOTFPerturber()
