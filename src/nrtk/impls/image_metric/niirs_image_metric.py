"""Defines an ImageMetric implementation to calculate NIIRS using pyBSM sensor and scenario configs.

Classes:
    NIIRSImageMetric: Computes the NIIRS metric using the defined sensor and scenario.

Dependencies:
    - pybsm.metrics.niirs5
    - pybsm.simulation.scenario.Scenario
    - pybsm.simulation.sensor.Sensor

Example usage:
    niirs_metric = NIIRSImageMetric(...)
    result = niirs_metric.compute_metric(image)
"""

from __future__ import annotations

__all__ = ["NIIRSImageMetric"]

from typing import Any

import numpy as np

from nrtk.interfaces.image_metric import ImageMetric
from nrtk.utils._exceptions import PyBSMImportError
from nrtk.utils._import_guard import import_guard

pybsm_available: bool = import_guard("pybsm", PyBSMImportError, ["metrics"])
from pybsm.metrics import niirs5  # noqa: E402
from pybsm.simulation.scenario import Scenario  # noqa: E402
from pybsm.simulation.sensor import Sensor  # noqa: E402
from typing_extensions import override  # noqa: E402

DEFAULT_PARAMETERS: dict[str, Any] = {
    "opt_trans_wavelengths": np.array([0.58 - 0.08, 0.58 + 0.08]) * 1.0e-6,
}


class NIIRSImageMetric(ImageMetric):
    """Implementation of the `ImageMetric` interface to calculate the NIIRS metric.

    The NIIRS metric, or National Imagery Interpretability Rating Scale, is used to rate
    the quality of images based on interpretability. This class requires sensor and
    scenario configurations to perform the metric calculation.

    Attributes:
        sensor (Sensor): The sensor configuration for the metric computation.
        scenario (Scenario): The scenario configuration used in metric calculation.
    """

    def __init__(
        self,
        sensor_name: str = "Sensor",
        D: float = 275e-3,  # noqa:N803
        f: float = 4,
        p_x: float = 0.008e-3,
        opt_trans_wavelengths: np.ndarray[Any, Any] = DEFAULT_PARAMETERS["opt_trans_wavelengths"],
        scenario_name: str = "Scenario",
        ihaze: int = 1,
        altitude: float = 9000,
        ground_range: float = 0,
        optics_transmission: np.ndarray[Any, Any] | None = None,
        eta: float = 0.0,
        w_x: float | None = None,
        w_y: float | None = None,
        int_time: float = 1.0,
        n_tdi: float = 1.0,
        dark_current: float = 0.0,
        read_noise: float = 0.0,
        max_n: int = int(100.0e6),
        bit_depth: float = 100.0,
        max_well_fill: float = 1.0,
        s_x: float = 0.0,
        s_y: float = 0.0,
        da_x: float = 0.0,
        da_y: float = 0.0,
        qe_wavelengths: np.ndarray[Any, Any] | None = None,
        qe: np.ndarray[Any, Any] | None = None,
        aircraft_speed: float = 0.0,
        target_reflectance: float = 0.15,
        target_temperature: float = 295.0,
        background_reflectance: float = 0.07,
        background_temperature: float = 293.0,
        ha_wind_speed: float = 21.0,
        cn2_at_1m: float = 1.7e-14,
        interp: bool = False,
    ) -> None:
        """Initializes the NIIRSImageMetric.

        Args:
            sensor_name:
                Name of the sensor.
            D:
                Effective aperture diameter (m).
            f:
                Focal length (m).
            p_x:
                Detector center-to-center spacings (pitch) in the x direction (meters).
            opt_trans_wavelengths:
                Specifies the spectral bandpass of the camera (m); at minimum, specify
                a start and end wavelength.
            scenario_name:
                Name of the scenario.
            ihaze:
                MODTRAN code for visibility, valid options are ihaze = 1 (Rural
                extinction with 23 km visibility) or ihaze = 2 (Rural extinction
                with 5 km visibility).
            altitude:
                Sensor height above ground level in meters.
            ground_range:
                Projection of line of sight between the camera and target along on the
                ground in meters.
            optics_transmission:
                Full system in-band optical transmission (unitless); do not include loss
                due to any telescope obscuration in this optical transmission array.
                Defaults to None.
            eta:
                Relative linear obscuration (unitless); obscuration of the aperture
                commonly occurs within telescopes due to secondary mirror or spider
                supports. Defaults to 0.0.
            w_x:
                Detector width in the x direction (m); if set equal to p_x,
                this corresponds to an assumed full pixel fill factor. Defaults to None.
            w_y:
                Detector width in the y direction (m); if set equal to p_x,
                this corresponds to an assumed full pixel fill factor. Defaults to None.
            int_time:
                Maximum integration time (s). Defaults to 1.0.
            n_tdi:
                Number of TDI stages (unitless). Defaults to 1.0.
            dark_current:
                Detector dark current (e-/s); dark current is the relatively small
                electric current that flows through photosensitive devices even when no
                photons enter the device. Defaults to 0.0.
            read_noise:
                Amount of noise generated by electronics as the charge present in the pixels
                (rms electrons). Defaults to 0.0.
            max_n:
                Detector electron well capacity (e-). Defaults to 100000000.
            bit_depth:
                Resolution of the detector ADC in bits (unitless). Defaults to 100.0.
            max_well_fill:
                Maximum amount of charge an individual pixel can hold before it
                becomes saturated. Defaults to 1.0.
            s_x:
                Root-mean-squared jitter amplitudes in the x direction (rad). Defaults to 0.0.
            s_y:
                Root-mean-squared jitter amplitudes in the y direction (rad). Defaults to 0.0.
            da_x:
                Line-of-sight angular drift rate during one integration time in the x
                direction (rad/s). Defaults to 0.0.
            da_y:
                Line-of-sight angular drift rate during one integration time in the y
                direction (rad/s). Defaults to 0.0.
            qe_wavelengths:
                Wavelengths corresponding to the array qe (m). Defaults to None.
            qe:
                Quantum efficiency as a function of wavelength (e-/photon). Defaults to None.
            aircraft_speed:
                Ground speed of the aircraft (m/s). Defaults to 0.0.
            target_reflectance:
                Object reflectance (unitless); the default 0.15 is the GIQE standard.
                Defaults to 0.15.
            target_temperature:
                Object temperature (Kelvin); 282 K is used for GIQE calculation.
                Defaults to 295.0.
            background_reflectance:
                Background reflectance (unitless). Defaults to 0.07.
            background_temperature:
                Background temperature (Kelvin); 280 K used for GIQE calculation.
                Defaults to 293.0.
            ha_wind_speed:
                The high altitude wind speed (m/s) used to calculate the turbulence
                profile; the default, 21.0, is the HV 5/7 profile value. Defaults to 21.0.
            cn2_at_1m:
                The refractive index structure parameter "near the ground"
                (e.g. at h = 1 m) used to calculate the turbulence profile; the
                default, 1.7e-14, is the HV 5/7 profile value. Defaults to 1.7e-14.
            interp:
                A flag to indicate whether atmospheric interpolation should be used.
                Defaults to False.

        Raises:
            PyBSMImportError: If the pyBSM library is not available.
        """
        if not self.is_usable():
            raise PyBSMImportError

        self.sensor: Sensor = Sensor(
            name=sensor_name,
            D=D,
            f=f,
            p_x=p_x,
            opt_trans_wavelengths=opt_trans_wavelengths,
            eta=eta,
            w_x=w_x,
            w_y=w_y,
            int_time=int_time,
            dark_current=dark_current,
            read_noise=read_noise,
            max_n=max_n,
            max_well_fill=max_well_fill,
            bit_depth=bit_depth,
            n_tdi=n_tdi,
            s_x=s_x,
            s_y=s_y,
            da_x=da_x,
            da_y=da_y,
        )
        self.scenario: Scenario = Scenario(
            name=scenario_name,
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

        if optics_transmission is not None:
            self.sensor.optics_transmission = optics_transmission

        if qe is not None:
            self.sensor.qe = qe

        if qe_wavelengths is not None:
            self.sensor.qe_wavelengths = qe_wavelengths

    @override
    def compute(
        self,
        img_1: np.ndarray[Any, Any] | None = None,
        img_2: np.ndarray[Any, Any] | None = None,
        additional_params: dict[str, Any] | None = None,
    ) -> float:
        """Given the Sensor and the Scenario, compute the NIIRS metric.

        Args:
            img_1: unused
            img_2: unused
            additional_params: unused

        While this implementation of compute() takes the expected input paramerters, none
        of the values are used during calculation. pyBSM's NIIRS calculation only uses
        the Sensor and Scenario objects to calculate NIIRS and is image independent.

        In order to inherit and function as an ImageMetric implementation, the arguements
        for compute stay consistent with other implementaiotns and are not used.

        Returns:
            The NIIRS metric for the given Sensor and Scenario.
        """
        # type ignore for pyright's handling of guarded import
        metrics = niirs5(sensor=self.sensor, scenario=self.scenario)  # type: ignore
        return metrics.niirs

    @override
    def __call__(
        self,
        img_1: np.ndarray[Any, Any] | None = None,
        img_2: np.ndarray[Any, Any] | None = None,
        additional_params: dict[str, Any] | None = None,
    ) -> float:
        """Given the Sensor and the Scenario, compute the NIIRS metric.

        Args:
            img_1: unused
            img_2: unused
            additional_params: unused

        While this implementation of compute() takes the expected input paramerters, none
        of the values are used during calculation. pyBSM's NIIRS calculation only uses
        the Sensor and Scenario objects to calculate NIIRS and is image independent.

        In order to inherit and function as an ImageMetric implementation, the arguements
        for compute stay consistent with other implementaiotns and are not used.

        Returns:
            The NIIRS metric for the given Sensor and Scenario.
        """
        return self.compute()

    @override
    def get_config(self) -> dict[str, Any]:
        """Generates a configuration dictionary for the NIIRSImageMetric instance."""
        return {
            "sensor_name": self.sensor.name,
            "D": self.sensor.D,
            "f": self.sensor.f,
            "p_x": self.sensor.p_x,
            "opt_trans_wavelengths": self.sensor.opt_trans_wavelengths,
            "scenario_name": self.scenario.name,
            "ihaze": self.scenario.ihaze,
            "altitude": self.scenario.altitude,
            "ground_range": self.scenario.ground_range,
            "optics_transmission": self.sensor.optics_transmission,
            "eta": self.sensor.eta,
            "w_x": self.sensor.w_x,
            "w_y": self.sensor.w_y,
            "int_time": self.sensor.int_time,
            "n_tdi": self.sensor.n_tdi,
            "dark_current": self.sensor.dark_current,
            "read_noise": self.sensor.read_noise,
            "max_n": self.sensor.max_n,
            "bit_depth": self.sensor.bit_depth,
            "max_well_fill": self.sensor.max_well_fill,
            "s_x": self.sensor.s_x,
            "s_y": self.sensor.s_y,
            "da_x": self.sensor.da_x,
            "da_y": self.sensor.da_y,
            "qe_wavelengths": self.sensor.qe_wavelengths,
            "qe": self.sensor.qe,
            "aircraft_speed": self.scenario.aircraft_speed,
            "target_reflectance": self.scenario.target_reflectance,
            "target_temperature": self.scenario.target_temperature,
            "background_reflectance": self.scenario.background_reflectance,
            "background_temperature": self.scenario.background_temperature,
            "ha_wind_speed": self.scenario.ha_wind_speed,
            "cn2_at_1m": self.scenario.cn2_at_1m,
            "interp": self.scenario._interp,  # noqa: SLF001
        }

    @classmethod
    def is_usable(cls) -> bool:
        """Returns True if the required pybsm module is available."""
        return pybsm_available
