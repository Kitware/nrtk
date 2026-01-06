"""Implements PybsmPerturber for image perturbations using pyBSM with sensor and scenario configs.

Classes:
    PybsmPerturber: Applies image perturbations using pyBSM based on specified sensor and
    scenario configurations.

Dependencies:
    - pyBSM for OTF-related functionalities.
    - nrtk.impls.perturb_image.optical.pybsm_otf_perturber.PybsmOTFPerturber for base functionality.

Example usage:
    >>> if not PybsmPerturber.is_usable():
    ...     import pytest
    ...
    ...     pytest.skip("PybsmPerturber is not usable")
    >>> sensor_and_scenario = {"f": 4, "altitude": 9000}
    >>> perturber = PybsmPerturber(**sensor_and_scenario)
    >>> image = np.ones((256, 256, 3))
    >>> img_gsd = 3.19 / 160
    >>> perturbed_image, _ = perturber.perturb(image=image, img_gsd=img_gsd)  # doctest: +SKIP
"""

from __future__ import annotations

__all__ = ["PybsmPerturber"]

from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.impls.perturb_image.optical.pybsm_otf_perturber import PybsmOTFPerturber
from nrtk.utils._exceptions import PyBSMImportError
from nrtk.utils._import_guard import import_guard

# Import checks
pybsm_available: bool = import_guard(module_name="pybsm", exception=PyBSMImportError, submodules=["simulation"])

from pybsm.simulation import ImageSimulator, SystemOTFSimulator  # noqa: E402

DEFAULT_PARAMETERS: dict[str, Any] = {
    "reflectance_range": np.array([0.05, 0.5]),  # It is bad standards to call np.array within argument defaults
    "opt_trans_wavelengths": np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6,
}


class PybsmPerturber(PybsmOTFPerturber):
    """Implements image perturbation using pyBSM sensor and scenario configurations.

    The `PybsmPerturber` class applies realistic perturbations to images by leveraging
    pyBSM's simulation functionalities. It takes in a sensor and scenario, along with
    other optional parameters, to simulate environmental effects on the image.

    See https://pybsm.readthedocs.io/en/latest/explanation.html for image formation concepts and parameter details.

    Attributes:
        reflectance_range (np.ndarray):
            Default reflectance range for image simulation.
    """

    def __init__(
        self,
        *,
        reflectance_range: np.ndarray[Any, Any] = DEFAULT_PARAMETERS["reflectance_range"],
        rng_seed: int | None = 1,
        sensor_name: str = "Sensor",
        D: float = 275e-3,  # noqa:N803
        f: float = 4,
        p_x: float = 0.008e-3,
        p_y: float | None = None,  # Defaults to None since the default value is dependent on p_x
        opt_trans_wavelengths: np.ndarray[Any, Any] = DEFAULT_PARAMETERS["opt_trans_wavelengths"],
        optics_transmission: np.ndarray[Any, Any]
        | None = None,  # Defaults to None since the default value is dependent on opt_trans_wavelengths
        eta: float = 0.0,
        w_x: float | None = None,  # Defaults to None since the default value is dependent on p_x
        w_y: float | None = None,  # Defaults to None since the default value is dependent on p_x
        int_time: float = 1.0,
        n_tdi: float = 1.0,
        dark_current: float = 0.0,
        read_noise: float = 0.0,
        max_n: int = int(100.0e6),
        bit_depth: float = 100.0,
        max_well_fill: float = 1.0,
        s_x: float = 0.0,
        s_y: float = 0.0,
        qe_wavelengths: np.ndarray[Any, Any]
        | None = None,  # Defaults to None since the default value is dependent on opt_trans_wavelengths
        qe: np.ndarray[Any, Any]
        | None = None,  # Defaults to None since the default value is dependent on opt_trans_wavelengths
        scenario_name: str = "Scenario",
        ihaze: int = 1,
        altitude: float = 9000,
        ground_range: float = 0,
        aircraft_speed: float = 0.0,
        target_reflectance: float = 0.15,
        target_temperature: float = 295.0,
        background_reflectance: float = 0.07,
        background_temperature: float = 293.0,
        ha_wind_speed: float = 21.0,
        cn2_at_1m: float = 1.7e-14,
        interp: bool = True,
    ) -> None:
        """Initializes the PybsmPerturber.

        Args:
            reflectance_range:
                Array of reflectances that correspond to pixel values.
            rng_seed:
                Random seed for reproducible results. Defaults to 1 for deterministic behavior.
            sensor_name:
                name of the sensor
            D:
                effective aperture diameter (m)
            f:
                focal length (m)
            p_x:
                detector center-to-center spacings (pitch) in the x and y directions
                (meters); if p_y is not provided, it is assumed equal to p_x
            opt_trans_wavelengths:
                specifies the spectral bandpass of the camera (m); at minimum, specify
                a start and end wavelength
            optics_transmission:
                full system in-band optical transmission (unitless); do not include loss
                due to any telescope obscuration in this optical transmission array
            eta:
                relative linear obscuration (unitless); obscuration of the aperture
                commonly occurs within telescopes due to secondary mirror or spider
                supports
            p_y:
                detector center-to-center spacings (pitch) in the x and y directions
                (meters); if p_y is not provided, it is assumed equal to p_x
            w_x:
                detector width in the x and y directions (m); if set equal to p_x and
                p_y, this corresponds to an assumed full pixel fill factor. In general,
                w_x and w_y are less than p_x and p_y due to non-photo-sensitive area
                (typically transistors) around each pixel.
            w_y:
                detector width in the x and y directions (m); if set equal to p_x and
                p_y, this corresponds to an assumed full pixel fill factor. In general,
                w_x and w_y are less than p_x and p_y due to non-photo-sensitive area
                (typically transistors) around each pixel.
            int_time:
                maximum integration time (s)
            qe:
                quantum efficiency as a function of wavelength (e-/photon)
            qe_wavelengths:
                wavelengths corresponding to the array qe (m)
            dark_current:
                detector dark current (e-/s); dark current is the relatively small
                electric current that flows through photosensitive devices even when no
                photons enter the device
            read_noise:
                amount of noise generated by electronics as the charge present in the pixels
            max_n:
                detector electron well capacity (e-); the default 100 million
                initializes to a large number so that, in the absence of better
                information, it doesn't affect outcomes
            bit_depth:
                resolution of the detector ADC in bits (unitless); default of 100 is a
                sufficiently large number so that in the absence of better information,
                it doesn't affect outcomes
            n_tdi:
                number of TDI stages (unitless)
            max_well_fill:
                maximum amount of charge an individual pixel can hold before it
                becomes saturated
            s_x:
                root-mean-squared jitter amplitudes in the x direction (rad)
            s_y:
                root-mean-squared jitter amplitudes in the y direction (rad)
            scenario_name:
                name of the scenario
            ihaze:
                MODTRAN code for visibility, valid options are ihaze = 1 (Rural
                extinction with 23 km visibility) or ihaze = 2 (Rural extinction
                with 5 km visibility)
            altitude:
                sensor height above ground level in meters; the database includes the
                following altitude options: 2 32.55 75 150 225 500 meters, 1000 to
                12000 in 1000 meter steps, and 14000 to 20000 in 2000 meter steps,
                24500
            ground_range:
                projection of line of sight between the camera and target along on the
                ground in meters; the distance between the target and the camera is
                given by sqrt(altitude^2 + ground_range^2).
                The following ground ranges are included in the database at each
                altitude until the ground range exceeds the distance to the spherical
                earth horizon: 0 100 500 1000 to 20000 in 1000 meter steps, 22000 to
                80000 in 2000 m steps, and  85000 to 300000 in 5000 meter steps.
            aircraft_speed:
                ground speed of the aircraft (m/s)
            target_reflectance:
                object reflectance (unitless); the default 0.15 is the giqe standard
            target_temperature:
                object temperature (Kelvin); 282 K is used for GIQE calculation
            background_reflectance:
                background reflectance (unitless)
            background_temperature:
                background temperature (Kelvin); 280 K used for GIQE calculation
            ha_wind_speed:
                the high altitude wind speed (m/s) used to calculate the turbulence
                profile; the default, 21.0, is the HV 5/7 profile value
            cn2_at_1m:
                the refractive index structure parameter "near the ground"
                (e.g. at h = 1 m) used to calculate the turbulence profile; the
                default, 1.7e-14, is the HV 5/7 profile value
            interp:
                A flag to indicate whether atmospheric interpolation should be used.
                Defaults to False.



        Raises:
            ImportError: If pyBSM is not found, install via `pip install nrtk[pybsm]`.
            ValueError: If reflectance_range length != 2
            ValueError: If reflectance_range not strictly ascending
        """
        if reflectance_range.shape[0] != 2:
            raise ValueError(f"Reflectance range array must have length of 2, got {reflectance_range.shape[0]}")
        if reflectance_range[0] >= reflectance_range[1]:
            raise ValueError(f"Reflectance range array values must be strictly ascending, got {reflectance_range}")

        # Initialize base class (which handles kwargs application to sensor/scenario)
        super().__init__(
            sensor_name=sensor_name,
            D=D,
            f=f,
            p_x=p_x,
            p_y=p_y,
            opt_trans_wavelengths=opt_trans_wavelengths,
            optics_transmission=optics_transmission,
            eta=eta,
            w_x=w_x,
            w_y=w_y,
            int_time=int_time,
            n_tdi=n_tdi,
            dark_current=dark_current,
            read_noise=read_noise,
            max_n=max_n,
            bit_depth=bit_depth,
            max_well_fill=max_well_fill,
            s_x=s_x,
            s_y=s_y,
            qe_wavelengths=qe_wavelengths,
            qe=qe,
            scenario_name=scenario_name,
            ihaze=ihaze,
            altitude=altitude,
            ground_range=ground_range,
            aircraft_speed=aircraft_speed,
            target_reflectance=target_reflectance,
            target_temperature=target_temperature,
            background_reflectance=background_reflectance,
            background_temperature=background_temperature,
            ha_wind_speed=ha_wind_speed,
            cn2_at_1m=cn2_at_1m,
            interp=interp,
        )

        # Store perturber-specific overrides
        self._rng = rng_seed
        self._reflectance_range: np.ndarray[Any, Any] = reflectance_range

        self._simulator = self._create_simulator()

    @override
    def _create_simulator(self) -> ImageSimulator:
        """Create SystemOTFSimulator with explicit parameters."""
        return SystemOTFSimulator(
            sensor=self.sensor,
            scenario=self.scenario,
            add_noise=True,
            rng=self._rng,
            use_reflectance=True,
            reflectance_range=self._reflectance_range,
        )

    def __str__(self) -> str:
        """Returns a string representation combining sensor and scenario names."""
        return self.sensor.name + " " + self.scenario.name

    def __repr__(self) -> str:
        """Returns a representation of the perturber including sensor and scenario names."""
        return self.__str__()

    @classmethod
    @override
    def from_config(cls, config_dict: dict[str, Any], merge_default: bool = True) -> PybsmPerturber:
        """Instantiates a PybsmPerturber from a configuration dictionary.

        Args:
            config_dict:
                Configuration dictionary with initialization parameters.
            merge_default:
                Whether to merge with default configuration. Defaults to True.

        Returns:
            An instance of PybsmPerturber.
        """
        config_dict = dict(config_dict)

        # # Convert input data to expected constructor types
        config_dict["reflectance_range"] = np.array(config_dict["reflectance_range"])
        config_dict["opt_trans_wavelengths"] = np.array(config_dict["opt_trans_wavelengths"])
        config_dict["optics_transmission"] = np.array(config_dict["optics_transmission"])
        config_dict["qe_wavelengths"] = np.array(config_dict["qe_wavelengths"])
        config_dict["qe"] = np.array(config_dict["qe"])

        return super(PybsmOTFPerturber, cls).from_config(config_dict, merge_default=merge_default)

    @override
    def get_config(self) -> dict[str, Any]:
        """Get current configuration including perturber-specific parameters."""
        cfg = super().get_config()
        cfg["reflectance_range"] = self._reflectance_range.tolist()
        cfg["rng_seed"] = self._rng

        return cfg

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Retrieves the default configuration for PybsmPerturber instances."""
        cfg = super().get_default_config()
        cfg["opt_trans_wavelengths"] = cfg["opt_trans_wavelengths"].tolist()
        cfg["reflectance_range"] = cfg["reflectance_range"].tolist()
        return cfg

    def _handle_boxes_and_format(
        self,
        *,
        sim_img: np.ndarray,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None,
        orig_shape: tuple,
    ) -> tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Override to normalize and handle box rescaling and format conversion to uint8."""
        sim = sim_img
        smin, smax = float(sim.min()), float(sim.max())
        if smax > smin:
            sim = (sim - smin) / (smax - smin) * 255.0
        # Convert to uint8
        sim_img_uint8 = sim.astype(np.uint8)

        # Rescale boxes if provided
        if boxes:
            scaled_boxes = self._rescale_boxes(boxes=boxes, orig_shape=orig_shape, new_shape=sim_img.shape)
            return sim_img_uint8, scaled_boxes

        return sim_img_uint8, boxes
