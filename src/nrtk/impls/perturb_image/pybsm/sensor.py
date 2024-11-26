"""
This module provides a wrapper for the pybsm.sensor module, enabling convenient creation and
configuration of sensor instances with customizable parameters. The primary class, `PybsmSensor`,
extends the functionality of `pybsm.sensor.Sensor` and integrates with the SMQTK framework.

Typical usage example:

    sensor = PybsmSensor(name="example",
                         D=0.3,
                         f=1.1,
                         p_x=0.4,
                         opt_trans_wavelengths=np.array([0.1, 0.4])*1.0e-6)
    out = sensor.create_sensor()

Attributes:
    pybsm_available (bool): Indicates if the pybsm module is available for use.

"""

from __future__ import annotations

from typing import Any, TypeVar

import numpy as np

try:
    from pybsm.simulation.sensor import Sensor

    pybsm_available = True
except ImportError:
    pybsm_available = False

from smqtk_core import Configurable

C = TypeVar("C", bound="PybsmSensor")


class PybsmSensor(Configurable):
    """
    Wrapper for pybsm.sensor.Sensor.

    This class allows for creating a sensor instance with specified parameters, managing
    sensor configurations, and enabling flexible integration into larger simulation frameworks.

    """

    def __init__(
        self,
        name: str,
        D: float,  # noqa:N803
        f: float,
        p_x: float,
        opt_trans_wavelengths: np.ndarray,
        optics_transmission: np.ndarray | None = None,
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
        qe_wavelengths: np.ndarray | None = None,
        qe: np.ndarray | None = None,
    ) -> None:
        """
        Initializes a PybsmSensor instance with specified configuration parameters.

        This is not intended to be a complete list but is more than adequate for the NIIRS demo (see
        pybsm.metrics.functional.niirs).

        :param name:
            name of the sensor
        :param D:
            effective aperture diameter (m)
        :param f:
            focal length (m)
        :param p_x:
            detector center-to-center spacings (pitch) in the x and y directions
            (meters); if p_y is not provided, it is assumed equal to p_x
        :param opt_trans_wavelengths:
            specifies the spectral bandpass of the camera (m); at minimum, specify
            a start and end wavelength
        :param optics_transmission:
            full system in-band optical transmission (unitless); do not include loss
            due to any telescope obscuration in this optical transmission array
        :param eta:
            relative linear obscuration (unitless); obscuration of the aperture
            commonly occurs within telescopes due to secondary mirror or spider
            supports
        :param p_y:
            detector center-to-center spacings (pitch) in the x and y directions
            (meters); if p_y is not provided, it is assumed equal to p_x
        :param w_x:
            detector width in the x and y directions (m); if set equal to p_x and
            p_y, this corresponds to an assumed full pixel fill factor. In general,
            w_x and w_y are less than p_x and p_y due to non-photo-sensitive area
            (typically transistors) around each pixel.
        :param w_y:
            detector width in the x and y directions (m); if set equal to p_x and
            p_y, this corresponds to an assumed full pixel fill factor. In general,
            w_x and w_y are less than p_x and p_y due to non-photo-sensitive area
            (typically transistors) around each pixel.
        :param int_time:
            maximum integration time (s)
        :param qe:
            quantum efficiency as a function of wavelength (e-/photon)
        :param qe_wavelengths:
            wavelengths corresponding to the array qe (m)
        :param other_irradiance:
            spectral irradiance from other sources (W/m^2 m); this is particularly
            useful for self emission in infrared cameras and may also represent
            stray light.
        :param dark_current:
            detector dark current (e-/s); dark current is the relatively small
            electric current that flows through photosensitive devices even when no
            photons enter the device
        :param max_n:
            detector electron well capacity (e-); the default 100 million
            initializes to a large number so that, in the absence of better
            information, it doesn't affect outcomes
        :param bit_depth:
            resolution of the detector ADC in bits (unitless); default of 100 is a
            sufficiently large number so that in the absence of better information,
            it doesn't affect outcomes
        :param n_tdi:
            number of TDI stages (unitless)
        :param cold_shield_temperature:
            temperature of the cold shield (K); it is a common approximation to
            assume that the coldshield is at the same temperature as the detector
            array
        :param optics_temperature:
            temperature of the optics (K)
        :param optics_emissivity:
            emissivity of the optics (unitless) except for the cold filter;
            a common approximation is 1-optics transmissivity
        :param cold_filter_transmission:
            transmission through the cold filter (unitless)
        :param cold_filter_temperature:
            temperature of the cold filter; it is a common approximation to assume
            that the filter is at the same temperature as the detector array
        :param cold_filter_emissivity:
            emissivity through the cold filter (unitless); a common approximation
            is 1-cold filter transmission
        :param s_x:
            root-mean-squared jitter amplitudes in the x direction (rad)
        :param s_y:
            root-mean-squared jitter amplitudes in the y direction (rad)
        :param da_x:
            line-of-sight angular drift rate during one integration time in the x
            direction (rad/s)
        :param da_y:
            line-of-sight angular drift rate during one integration time in the y
            direction (rad/s)
        :param pv:
            wavefront error phase variance (rad^2) -- tip: write as (2*pi*waves of
            error)^2
        :param pv_wavelength:
            wavelength at which pv is obtained (m)

        """
        self._check_opt_trans_wavelengths(opt_trans_wavelengths)

        # required parameters
        self.name = name
        self.D = D
        self.f = f
        self.p_x = p_x
        self.opt_trans_wavelengths = opt_trans_wavelengths

        # optional parameters
        self._set_optics_transmission(optics_transmission=optics_transmission)
        self.eta = eta
        self.p_y = p_x
        self.w_x = p_x if w_x is None else w_x
        self.w_y = p_x if w_y is None else w_y
        self.int_time = int_time
        self.n_tdi = n_tdi
        self.dark_current = dark_current
        self.read_noise = read_noise
        self.max_n = max_n
        self.max_well_fill = max_well_fill
        self.bit_depth = bit_depth
        if qe_wavelengths is None:
            self.qe_wavelengths = opt_trans_wavelengths
        else:
            self.qe_wavelengths = qe_wavelengths
        if qe is None:
            self.qe = np.ones(opt_trans_wavelengths.shape[0])
        else:
            self.qe = qe

        # not yet added to constructor
        self.other_irradiance = 0.0
        self.cold_shield_temperature = 70.0
        self.optics_temperature = 270.0
        self.optics_emissivity = 0.0
        self.cold_filter_transmission = 1.0
        self.cold_filter_temperature = 70.0
        self.cold_filter_emissivity = 0.0
        self.s_x = s_x
        self.s_y = s_y
        self.da_x = da_x
        self.da_y = da_y
        self.pv = 0.0
        self.pv_wavelength = 0.633e-6  # typical value
        self.L_x = D
        self.L_y = D
        self.other_noise = np.array([0])
        self.filter_kernel = np.array([1])
        self.frame_stacks = 1

    def _check_opt_trans_wavelengths(self, opt_trans_wavelengths: np.ndarray) -> None:
        """
        Validates the `opt_trans_wavelengths` array to ensure it meets the required criteria.

        This method checks that the `opt_trans_wavelengths` array has at least two elements,
        representing the start and end wavelengths, and that the wavelengths are in ascending order.

        Parameters
        ----------
        opt_trans_wavelengths : np.ndarray
            An array of optical transmission wavelengths. The array must contain at least
            two elements and must be in ascending order.

        Raises
        ------
        ValueError
            If `opt_trans_wavelengths` contains fewer than two elements.
        ValueError
            If the wavelengths in `opt_trans_wavelengths` are not in ascending order.

        Returns
        -------
        None
            This method does not return any value; it only performs validation.
        """
        if opt_trans_wavelengths.shape[0] < 2:
            raise ValueError(
                "At minimum, at least the start and end wavelengths must be specified for opt_trans_wavelengths",
            )
        if opt_trans_wavelengths[0] >= opt_trans_wavelengths[-1]:
            raise ValueError("opt_trans_wavelengths must be ascending")

    def _set_optics_transmission(
        self,
        optics_transmission: np.ndarray | None = None,
    ) -> None:
        """
        This method assigns the `optics_transmission` array to the instance. If no array is provided,
        it initializes `optics_transmission` to an array of ones with the same length as `opt_trans_wavelengths`.
        It ensures that the provided `optics_transmission` array matches the length of `opt_trans_wavelengths`.

        Parameters
        ----------
        optics_transmission : np.ndarray or None, optional
            An array representing the optics transmission values corresponding to `opt_trans_wavelengths`.
            If None, the optics transmission is set to an array of ones. The array must have the same
            length as `opt_trans_wavelengths` if provided.

        Raises
        ------
        ValueError
            If `optics_transmission` is provided and its length does not match the length of
            `opt_trans_wavelengths`.

        Returns
        -------
        None
            This method does not return any value; it sets the `optics_transmission` attribute.
        """
        if optics_transmission is None:
            self.optics_transmission = np.ones(self.opt_trans_wavelengths.shape[0])
        else:
            if optics_transmission.shape[0] != self.opt_trans_wavelengths.shape[0]:
                raise ValueError("optics_transmission and opt_trans_wavelengths must have the same length")
            self.optics_transmission = optics_transmission

    def __str__(self) -> str:
        """
        Returns the provided name as the string representation

        Returns:
            str: name of instance
        """
        return self.name

    def __repr__(self) -> str:
        """
        Returns the provided name as the object representation

        Returns:
            str: name of instance
        """
        return self.name

    def create_sensor(self) -> Sensor:
        """
        Initializes and returns a pybsm.sensor.Sensor instance based on the current configuration.

        Returns:
            Sensor: A configured instance of pybsm.sensor.Sensor, if pybsm is available.

        Raises:
            ImportError: If the pybsm module is unavailable.
            ValueError: If configuration parameters are invalid.

        Example:
            >>> sensor = PybsmSensor(
            ...     name="example", D=0.3, f=1.1, p_x=0.4, opt_trans_wavelengths=np.array([0.1, 0.4]) * 1.0e-6
            ... )
            >>> sensor_instance = sensor.create_sensor()

        """
        if not self.is_usable():
            raise ImportError("pybsm not found")
        S = Sensor(self.name, self.D, self.f, self.p_x, self.opt_trans_wavelengths)  # noqa:N806
        S.optics_transmission = self.optics_transmission
        S.eta = self.eta
        S.p_y = self.p_y
        S.w_x = self.w_x
        S.w_y = self.w_y
        S.int_time = self.int_time
        S.n_tdi = self.n_tdi
        S.dark_current = self.dark_current
        S.read_noise = self.read_noise
        S.max_n = self.max_n
        S.max_well_fill = self.max_well_fill
        S.bit_depth = self.bit_depth
        S.qe_wavelengths = self.qe_wavelengths
        S.qe = self.qe
        return S

    def __call__(self) -> Sensor:
        """Alias for :meth:`.StoreSensor.sensor`."""
        return self.create_sensor()

    @classmethod
    def from_config(cls: type[C], config_dict: dict, merge_default: bool = True) -> C:
        """
        Rehydrates an object instance from a serializable config dictionary

        Args:
            cls (type[C]): The class of the object which will be instantiated
            config_dict (dict): Dictionary of serializable values that will be
                                included in the object instance
            merge_default (bool, optional): Indicator variable describing whether
                                            or not to use default config values.
                                            Defaults to True.

        Returns:
            C: Instantiation of class of type C
        """
        config_dict = dict(config_dict)

        # Convert input data to expected constructor types
        config_dict["opt_trans_wavelengths"] = np.array(config_dict["opt_trans_wavelengths"])

        # Non-JSON type arguments with defaults (so they might not be there)
        optics_transmission = config_dict.get("optics_transmission", None)
        if optics_transmission is not None:
            config_dict["optics_transmission"] = np.array(config_dict["optics_transmission"])
        qe_wavelengths = config_dict.get("qe_wavelengths", None)
        if qe_wavelengths is not None:
            config_dict["qe_wavelengths"] = np.array(config_dict["qe_wavelengths"])
        qe = config_dict.get("qe", None)
        if qe is not None:
            config_dict["qe"] = np.array(config_dict["qe"])

        return super().from_config(config_dict, merge_default=merge_default)

    def get_config(self) -> dict[str, Any]:
        """
        Generates a serializable config that can be used to rehydrate object

        Returns:
            dict[str, Any]: serializable config containing all instance parameters
        """
        return {
            "name": self.name,
            "D": self.D,
            "f": self.f,
            "p_x": self.p_x,
            "opt_trans_wavelengths": self.opt_trans_wavelengths.tolist(),
            "optics_transmission": self.optics_transmission.tolist(),
            "eta": self.eta,
            "w_x": self.w_x,
            "w_y": self.w_y,
            "int_time": self.int_time,
            "n_tdi": self.n_tdi,
            "dark_current": self.dark_current,
            "read_noise": self.read_noise,
            "max_n": self.max_n,
            "bit_depth": self.bit_depth,
            "max_well_fill": self.max_well_fill,
            "s_x": self.s_x,
            "s_y": self.s_y,
            "da_x": self.da_x,
            "da_y": self.da_y,
            "qe_wavelengths": self.qe_wavelengths.tolist(),
            "qe": self.qe.tolist(),
        }

    @classmethod
    def is_usable(cls) -> bool:
        """
        Indicates if the pybsm module is available and usable.

        Returns:
            bool: True if pybsm is installed and accessible, False otherwise.
        """
        return pybsm_available
