from __future__ import annotations

from typing import Any, Dict, Optional, Type, TypeVar

import numpy as np
from pybsm.simulation.sensor import Sensor
from smqtk_core import Configurable

C = TypeVar("C", bound="PybsmSensor")


class PybsmSensor(Configurable):
    """Wrapper for pybsm.sensor.

    Attributes (the first four are mandatory):
    ------------------------------------------
    name :
        name of the sensor (string)
    D :
        effective aperture diameter (m)
    f :
        focal length (m)
    p_x and p_y :
        detector center-to-center spacings (pitch) in the x and y directions (m)
    opt_trans_wavelengths :
        numpy array specifying the spectral bandpass of the camera (m).  At
        minimum, start and end wavelength should be specified.
    optics_transmission :
        full system in-band optical transmission (unitless).  Loss due to any
        telescope obscuration should *not* be included in with this optical transmission
        array.
    eta :
        relative linear obscuration (unitless)
    w_x and w_y :
        detector width in the x and y directions (m)
    int_time :
        maximum integration time (s)
    n_tdi:
        the number of time-delay integration stages (relevant only when TDI cameras
        are used. For CMOS cameras, the value can be assumed to be 1.0)
    qe :
        quantum efficiency as a function of wavelength (e-/photon)
    qe_wavelengths :
        wavelengths corresponding to the array qe (m)
    other_irradiance :
        spectral irradiance from other sources (W/m^2 m).
        This is particularly useful for self emission in infrared cameras.  It may
        also represent stray light.
    dark_current :
        detector dark current (e-/s)
    max_n :
        detector electron well capacity (e-)
    max_well_fill :
        desired well fill, i.e. Maximum well size x Desired fill fraction
    bit_depth :
        resolution of the detector ADC in bits (unitless)
    cold_shield_temperature :
        temperature of the cold shield (K).  It is a common approximation to assume
        that the coldshield is at the same temperature as the detector array.
    optics_temperature :
        temperature of the optics (K)
    optics_emissivity :
        emissivity of the optics (unitless) except for the cold filter.
        A common approximation is 1-optics transmissivity.
    cold_filter_transmission :
        transmission through the cold filter (unitless)
    cold_filter_temperature :
        temperature of the cold filter.  It is a common approximation to assume
        that the filter is at the same temperature as the detector array.
    cold_filter_emissivity :
        emissivity through the cold filter (unitless).  A common approximation
        is 1-cold filter transmission
    s_x and s_y :
        Root-mean-squared jitter amplitudes in the x and y directions respectively. (rad)
    da_x and da_y :
        line-of-sight angular drift rate during one integration time in the x and y
        directions respectively. (rad/s)
    pv :
        wavefront error phase variance (rad^2) - tip: write as (2*pi*waves of error)^2
    pv_wavelength :
        wavelength at which pv is obtained (m)
    L_x and L_y :
        correlation lengths of the phase autocorrelation function.  Apparently,
        it is common to set the L_x and L_y to the aperture diameter.  (m)
    other_noise :
        a catch all for noise terms that are not explicitly included elsewhere
        (read noise, photon noise, dark current, quantization noise are
        all already included)
    filter_kernel:
         2-D filter kernel (for sharpening or whatever).  Note that
         the kernel is assumed to sum to one.
    frame_stacks:
         the number of frames to be added together for improved SNR.

    :raises: ValueError if opt_trans_wavelengths length < 2
    :raises: ValueError if opt_trans_wavelengths is not ascending
    :raises: ValueError if opt_trans_wavelengths and (if provided) optics_transmission lengths are different
    """

    def __init__(
        self,
        name: str,
        D: float,  # noqa:N803
        f: float,
        p_x: float,
        opt_trans_wavelengths: np.ndarray,
        optics_transmission: Optional[np.ndarray] = None,
        eta: float = 0.0,
        w_x: Optional[float] = None,
        w_y: Optional[float] = None,
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
        qe_wavelengths: Optional[np.ndarray] = None,
        qe: Optional[np.ndarray] = None,
    ):
        if opt_trans_wavelengths.shape[0] < 2:
            raise ValueError(
                "At minimum, at least the start and end wavelengths" " must be specified for opt_trans_wavelengths"
            )
        if opt_trans_wavelengths[0] >= opt_trans_wavelengths[-1]:
            raise ValueError("opt_trans_wavelengths must be ascending")

        # required parameters
        self.name = name
        self.D = D
        self.f = f
        self.p_x = p_x
        self.opt_trans_wavelengths = opt_trans_wavelengths

        # optional parameters
        if optics_transmission is None:
            self.optics_transmission = np.ones(opt_trans_wavelengths.shape[0])
        else:
            if optics_transmission.shape[0] != opt_trans_wavelengths.shape[0]:
                raise ValueError("optics_transmission and opt_trans_wavelengths must have the same length")
            self.optics_transmission = optics_transmission
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

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def create_sensor(self) -> Sensor:
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
    def from_config(cls: Type[C], config_dict: Dict, merge_default: bool = True) -> C:
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

    def get_config(self) -> Dict[str, Any]:
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
