import unittest.mock as mock
from contextlib import nullcontext as does_not_raise
from typing import Any, ContextManager, Dict, Optional, Sequence

import numpy as np
import pybsm.radiance as radiance
import pytest
from PIL import Image
from pybsm.utils import load_database_atmosphere
from smqtk_core.configuration import configuration_test_helper
from syrupy.assertion import SnapshotAssertion

from nrtk.impls.perturb_image.pybsm.turbulence_aperture_otf_perturber import (
    TurbulenceApertureOTFPerturber,
)

from ...test_pybsm_utils import (
    TIFFImageSnapshotExtension,
    create_sample_sensor_and_scenario,
)
from ..test_perturber_utils import pybsm_perturber_assertions

INPUT_IMG_FILE = "./examples/pybsm/data/M-41 Walker Bulldog (USA) width 319cm height 272cm.tiff"


@pytest.mark.skipif(
    not TurbulenceApertureOTFPerturber.is_usable(),
    reason="OpenCV not found. Please install 'nrtk[graphics]' or `nrtk[headless]`.",
)
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
        ),
        [
            (False, None, None, None, None, None, None, None, None, None, None),
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
            ),
        ],
    )
    def test_reproducibility(
        self,
        use_sensor_scenario: bool,
        mtf_wavelengths: Optional[Sequence[float]],
        mtf_weights: Optional[Sequence[float]],
        altitude: Optional[float],
        slant_range: Optional[float],
        D: Optional[float],  # noqa: N803
        ha_wind_speed: Optional[float],
        cn2_at_1m: Optional[float],
        int_time: Optional[float],
        n_tdi: Optional[float],
        aircraft_speed: Optional[float],
    ) -> None:
        """Ensure results are reproducible."""
        img = np.array(Image.open(INPUT_IMG_FILE))
        img_md = {"img_gsd": 3.19 / 160.0}

        sensor = None
        scenario = None
        if use_sensor_scenario:
            sensor, scenario = create_sample_sensor_and_scenario()

        inst = TurbulenceApertureOTFPerturber(
            sensor=sensor,
            scenario=scenario,
            mtf_wavelengths=mtf_wavelengths,
            mtf_weights=mtf_weights,
            altitude=altitude,
            slant_range=slant_range,
            D=D,  # noqa: N803
            ha_wind_speed=ha_wind_speed,
            cn2_at_1m=cn2_at_1m,
            int_time=int_time,
            n_tdi=n_tdi,
            aircraft_speed=aircraft_speed,
        )

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
        perturber = TurbulenceApertureOTFPerturber(sensor=sensor, scenario=scenario)
        img = np.array(Image.open(INPUT_IMG_FILE))
        with expectation:
            _ = perturber.perturb(img, additional_params)

    @pytest.mark.parametrize(
        ("mtf_wavelengths", "mtf_weights", "cn2_at_1m", "expectation"),
        [
            ([0.5e-6, 0.6e-6], [0.5, 0.5], 0.1, does_not_raise()),
            (
                [0.5e-6, 0.6e-6],
                [],
                0.1,
                pytest.raises(ValueError, match=r"mtf_weights is empty"),
            ),
            (
                [],
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
        expectation: ContextManager,
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
        ),
        [
            (False, None, None, None, None, None, None, None, None, None, None),
            (True, None, None, None, None, None, None, None, None, None, None),
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
            ),
            (
                False,
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
            ),
        ],
    )
    def test_configuration(
        self,
        use_sensor_scenario: bool,
        mtf_wavelengths: Optional[Sequence[float]],
        mtf_weights: Optional[Sequence[float]],
        altitude: Optional[float],
        slant_range: Optional[float],
        D: Optional[float],  # noqa: N803
        ha_wind_speed: Optional[float],
        cn2_at_1m: Optional[float],
        int_time: Optional[float],
        n_tdi: Optional[float],
        aircraft_speed: Optional[float],
    ) -> None:
        """Test configuration stability."""
        sensor = None
        scenario = None
        if use_sensor_scenario:
            sensor, scenario = create_sample_sensor_and_scenario()
            atm = load_database_atmosphere(scenario.altitude, scenario.ground_range, scenario.ihaze)
            _, _, spectral_weights = radiance.reflectance_to_photoelectrons(atm, sensor, sensor.int_time)

            wavelengths = spectral_weights[0]
            weights = spectral_weights[1]

            # cut down the wavelength range to only the regions of interests
            pos_weights = np.where(weights > 0.0)

        inst = TurbulenceApertureOTFPerturber(
            sensor=sensor,
            scenario=scenario,
            mtf_wavelengths=mtf_wavelengths,
            mtf_weights=mtf_weights,
            altitude=altitude,
            slant_range=slant_range,
            D=D,  # noqa: N803
            ha_wind_speed=ha_wind_speed,
            cn2_at_1m=cn2_at_1m,
            int_time=int_time,
            n_tdi=n_tdi,
            aircraft_speed=aircraft_speed,
        )

        for i in configuration_test_helper(inst):
            # MTF wavelengths and weights
            if mtf_wavelengths is not None:
                assert np.array_equal(i.mtf_wavelengths, mtf_wavelengths)
            elif sensor is not None and scenario is not None:
                assert np.array_equal(i.mtf_wavelengths, wavelengths[pos_weights])
            else:  # Default value
                assert np.allclose(i.mtf_wavelengths, np.array([0.50e-6, 0.66e-6]))
            if mtf_weights is not None:
                assert np.array_equal(i.mtf_weights, mtf_weights)
            elif sensor is not None and scenario is not None:
                assert np.array_equal(i.mtf_weights, weights[pos_weights])
            else:  # Default value
                assert np.allclose(i.mtf_weights, np.array([1.0, 1.0]))

            # Sensor parameters
            if D is not None:  # noqa: N806
                assert i.D == D  # noqa: N806
            elif sensor is not None and scenario is not None:
                assert i.D == sensor.D  # noqa: N806
            else:  # Default value
                assert i.D == 40e-3  # noqa: N806
            if int_time is not None:
                assert i.int_time == int_time
            elif sensor is not None and scenario is not None:
                assert i.int_time == sensor.int_time
            else:  # Default value
                assert i.int_time == 30e-3
            if n_tdi is not None:
                assert i.n_tdi == n_tdi
            elif sensor is not None and scenario is not None:
                assert i.n_tdi == sensor.n_tdi
            else:  # Default value
                assert i.n_tdi == 1.0

            # Scenario parameters
            if altitude is not None:
                assert i.altitude == altitude
            elif sensor is not None and scenario is not None:
                assert i.altitude == scenario.altitude
            else:  # Default value
                assert i.altitude == 250
            if ha_wind_speed is not None:
                assert i.ha_wind_speed == ha_wind_speed
            elif sensor is not None and scenario is not None:
                assert i.ha_wind_speed == scenario.ha_wind_speed
            else:  # Default value
                assert i.ha_wind_speed == 0
            if cn2_at_1m is not None:
                assert i.cn2_at_1m == cn2_at_1m
            elif sensor is not None and scenario is not None:
                assert i.cn2_at_1m == scenario.cn2_at_1m
            else:  # Default value
                assert i.cn2_at_1m == 1.7e-14
            if aircraft_speed is not None:
                assert i.aircraft_speed == aircraft_speed
            elif sensor is not None and scenario is not None:
                assert i.aircraft_speed == scenario.aircraft_speed
            else:  # Default value
                assert i.aircraft_speed == 0

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
                assert i.sensor.n_tdi == sensor.n_tdi
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
        ),
        [
            (False, None, None, None, None, None, None, None, None, None, None),
            (True, None, None, None, None, None, None, None, None, None, None),
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
            ),
            (
                False,
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
            ),
        ],
    )
    def test_regression(
        self,
        snapshot: SnapshotAssertion,
        use_sensor_scenario: bool,
        mtf_wavelengths: Optional[Sequence[float]],
        mtf_weights: Optional[Sequence[float]],
        altitude: Optional[float],
        slant_range: Optional[float],
        D: Optional[float],  # noqa: N803
        ha_wind_speed: Optional[float],
        cn2_at_1m: Optional[float],
        int_time: Optional[float],
        n_tdi: Optional[float],
        aircraft_speed: Optional[float],
    ) -> None:
        """Regression testing results to detect API changes."""
        img = np.array(Image.open(INPUT_IMG_FILE))
        img_md = {"img_gsd": 3.19 / 160.0}

        sensor = None
        scenario = None
        if use_sensor_scenario:
            sensor, scenario = create_sample_sensor_and_scenario()

        inst = TurbulenceApertureOTFPerturber(
            sensor=sensor,
            scenario=scenario,
            mtf_wavelengths=mtf_wavelengths,
            mtf_weights=mtf_weights,
            altitude=altitude,
            slant_range=slant_range,
            D=D,  # noqa: N803
            ha_wind_speed=ha_wind_speed,
            cn2_at_1m=cn2_at_1m,
            int_time=int_time,
            n_tdi=n_tdi,
            aircraft_speed=aircraft_speed,
        )

        out_img = pybsm_perturber_assertions(perturb=inst, image=img, expected=None, additional_params=img_md)

        assert TIFFImageSnapshotExtension.ndarray2bytes(out_img) == snapshot(extension_class=TIFFImageSnapshotExtension)


@mock.patch.object(TurbulenceApertureOTFPerturber, "is_usable")
def test_missing_deps(mock_is_usable) -> None:
    """Test that an exception is raised when required dependencies are not installed."""
    mock_is_usable.return_value = False
    assert not TurbulenceApertureOTFPerturber.is_usable()
    with pytest.raises(ImportError, match=r"OpenCV not found"):
        TurbulenceApertureOTFPerturber()