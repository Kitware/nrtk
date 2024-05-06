from __future__ import annotations
from typing import Any
from typing import Dict, Optional, Type

import numpy as np
from pybsm.simulation.sensor import Sensor
from smqtk_core import Configurable


class PybsmSensor(Configurable):
    """
    Wrapper for pybsm.sensor.

    Attributes (the first four are mandatory):
    ------------------------------------------
    name :
        name of the sensor (string)
    D :
        effective aperture diameter (m)
    f :
        focal length (m)
    px and py :
        detector center-to-center spacings (pitch) in the x and y directions (m)
    optTransWavelengths :
        numpy array specifying the spectral bandpass of the camera (m).  At
        minimum, start and end wavelength should be specified.
    opticsTransmission :
        full system in-band optical transmission (unitless).  Loss due to any
        telescope obscuration should *not* be included in with this optical transmission
        array.
    eta :
        relative linear obscuration (unitless)
    wx and wy :
        detector width in the x and y directions (m)
    qe :
        quantum efficiency as a function of wavelength (e-/photon)
    qewavelengths :
        wavelengths corresponding to the array qe (m)
    otherIrradiance :
        spectral irradiance from other sources (W/m^2 m).
        This is particularly useful for self emission in infrared cameras.  It may
        also represent stray light.
    darkCurrent :
        detector dark current (e-/s)
    maxN :
        detector electron well capacity (e-)
    maxFill :
        desired well fill, i.e. Maximum well size x Desired fill fraction
    bitdepth :
        resolution of the detector ADC in bits (unitless)
    ntdi :
        number of TDI stages (unitless)
    coldshieldTemperature :
        temperature of the cold shield (K).  It is a common approximation to assume
        that the coldshield is at the same temperature as the detector array.
    opticsTemperature :
        temperature of the optics (K)
    opticsEmissivity :
        emissivity of the optics (unitless) except for the cold filter.
        A common approximation is 1-optics transmissivity.
    coldfilterTransmission :
        transmission through the cold filter (unitless)
    coldfilterTemperature :
        temperature of the cold filter.  It is a common approximation to assume
        that the filter is at the same temperature as the detector array.
    coldfilterEmissivity :
        emissivity through the cold filter (unitless).  A common approximation
        is 1-cold filter transmission
    sx and sy :
        Root-mean-squared jitter amplitudes in the x and y directions respectively. (rad)
    dax and day :
        line-of-sight angular drift rate during one integration time in the x and y
        directions respectively. (rad/s)
    pv :
        wavefront error phase variance (rad^2) - tip: write as (2*pi*waves of error)^2
    pvwavelength :
        wavelength at which pv is obtained (m)
    Lx and Ly :
        correlation lengths of the phase autocorrelation function.  Apparently,
        it is common to set the Lx and Ly to the aperture diameter.  (m)
    otherNoise :
        a catch all for noise terms that are not explicitly included elsewhere
        (read noise, photon noise, dark current, quantization noise are
        all already included)
    filterKernel:
         2-D filter kernel (for sharpening or whatever).  Note that
         the kernel is assumed to sum to one.
    framestacks:
         the number of frames to be added together for improved SNR.

    :raises: ValueError if optTransWavelengths length < 2
    :raises: ValueError if optTransWavelengths is not ascending
    :raises: ValueError if optTransWavelengths and (if provided) opticsTransmission lengths are different
    """
    def __init__(self,
                 name: str,
                 D: float,
                 f: float,
                 px: float,
                 optTransWavelengths: np.ndarray,
                 opticsTransmission: Optional[np.ndarray] = None,
                 eta: Optional[float] = 0.0,
                 wx: Optional[float] = None,
                 wy: Optional[float] = None,
                 intTime: Optional[float] = 1.0,
                 darkCurrent: Optional[float] = 0.0,
                 readNoise: Optional[float] = 0.0,
                 maxN: Optional[float] = 100.0e6,
                 bitdepth: Optional[float] = 100.0,
                 maxWellFill: Optional[float] = 1.0,
                 sx: Optional[float] = 0.0,
                 sy: Optional[float] = 0.0,
                 dax: Optional[float] = 0.0,
                 day: Optional[float] = 0.0,
                 qewavelengths: Optional[np.ndarray] = None,
                 qe: Optional[np.ndarray] = None
                 ):
        if optTransWavelengths.shape[0] < 2:
            raise ValueError("At minimum, at least the start and end wavelengths"
                             " must be specified for optTransWavelengths")
        if optTransWavelengths[0] >= optTransWavelengths[-1]:
            raise ValueError("optTransWavelengths must be ascending")

        # required parameters
        self.name = name
        self.D = D
        self.f = f
        self.px = px
        self.optTransWavelengths = optTransWavelengths

        # optional parameters
        if opticsTransmission is None:
            self.opticsTransmission = np.ones(optTransWavelengths.shape[0])
        else:
            if opticsTransmission.shape[0] != optTransWavelengths.shape[0]:
                raise ValueError("opticsTransmission and optTransWavelengths must have the same length")
            self.opticsTransmission = opticsTransmission
        self.eta = eta
        self.py = px
        self.wx = px if wx is None else wx
        self.wy = px if wy is None else wy
        self.intTime = intTime
        self.darkCurrent = darkCurrent
        self.readNoise = readNoise
        self.maxN = maxN
        self.maxWellFill = maxWellFill
        self.bitdepth = bitdepth
        if qewavelengths is None:
            self.qewavelengths = optTransWavelengths
        else:
            self.qewavelengths = qewavelengths
        if qe is None:
            self.qe = np.ones(optTransWavelengths.shape[0])
        else:
            self.qe = qe

        # not yet added to constructor
        self.ntdi = 1.0
        self.otherIrradiance = 0.0
        self.coldshieldTemperature = 70.0
        self.opticsTemperature = 270.0
        self.opticsEmissivity = 0.0
        self.coldfilterTransmission = 1.0
        self.coldfilterTemperature = 70.0
        self.coldfilterEmissivity = 0.0
        self.sx = sx
        self.sy = sy
        self.dax = dax
        self.day = day
        self.pv = 0.0
        self.pvwavelength = .633e-6  # typical value
        self.Lx = D
        self.Ly = D
        self.otherNoise = np.array([0])
        self.filterKernel = np.array([1])
        self.framestacks = 1

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def create_sensor(self) -> Sensor:
        S = Sensor(self.name, self.D, self.f, self.px, self.optTransWavelengths)
        S.opticsTransmission = self.opticsTransmission
        S.eta = self.eta
        S.py = self.py
        S.wx = self.wx
        S.wy = self.wy
        S.intTime = self.intTime
        S.darkCurrent = self.darkCurrent
        S.readNoise = self.readNoise
        S.maxN = self.maxN
        S.maxWellFill = self.maxWellFill
        S.bitdepth = self.bitdepth
        S.qewavelengths = self.qewavelengths
        S.qe = self.qe
        return S

    def __call__(
        self
    ) -> Sensor:
        """
        Alias for :meth:`.StoreSensor.sensor`.
        """
        return self.create_sensor()

    @classmethod
    def from_config(
        cls: Type[PybsmSensor],
        config_dict: Dict,
        merge_default: bool = True
    ) -> PybsmSensor:
        config_dict = dict(config_dict)

        # Convert input data to expected constructor types
        config_dict["optTransWavelengths"] = np.array(config_dict["optTransWavelengths"])

        # Non-JSON type arguments with defaults (so they might not be there)
        opticsTransmission = config_dict.get("opticsTransmission", None)
        if opticsTransmission is not None:
            config_dict["opticsTransmission"] = np.array(config_dict["opticsTransmission"])
        qewavelengths = config_dict.get("qewavelengths", None)
        if qewavelengths is not None:
            config_dict["qewavelengths"] = np.array(config_dict["qewavelengths"])
        qe = config_dict.get("qe", None)
        if qe is not None:
            config_dict["qe"] = np.array(config_dict["qe"])

        return super().from_config(config_dict, merge_default=merge_default)

    def get_config(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'D': self.D,
            'f': self.f,
            'px': self.px,
            'optTransWavelengths': self.optTransWavelengths.tolist(),
            'opticsTransmission': self.opticsTransmission.tolist(),
            'eta': self.eta,
            'wx': self.wx,
            'wy': self.wy,
            'intTime': self.intTime,
            'darkCurrent': self.darkCurrent,
            'readNoise': self.readNoise,
            'maxN': self.maxN,
            'bitdepth': self.bitdepth,
            'maxWellFill': self.maxWellFill,
            'sx': self.sx,
            'sy': self.sy,
            'dax': self.dax,
            'day': self.day,
            'qewavelengths': self.qewavelengths.tolist(),
            'qe': self.qe.tolist(),
        }
