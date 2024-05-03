from typing import Any
from typing import Dict, Optional

from pybsm.simulation.scenario import Scenario
from smqtk_core import Configurable


class PybsmScenario(Configurable):
    """
    Wrapper for pybsm.scenario.

    ihaze:
        MODTRAN code for visibility, valid options are ihaze = 1 (Rural extinction with 23 km visibility)
        or ihaze = 2 (Rural extinction with 5 km visibility)
    altitude:
        sensor height above ground level in meters.  The database includes the following
        altitude options: 2 32.55 75 150 225 500 meters, 1000 to 12000 in 1000 meter steps,
        and 14000 to 20000 in 2000 meter steps, 24500
    groundRange:
        distance *on the ground* between the target and sensor in meters.
        The following ground ranges are included in the database at each altitude
        until the ground range exceeds the distance to the spherical earth horizon:
        0 100 500 1000 to 20000 in 1000 meter steps, 22000 to 80000 in 2000 m steps,
        and  85000 to 300000 in 5000 meter steps.
    aircraftSpeed:
        ground speed of the aircraft (m/s)
    targetReflectance:
        object reflectance (unitless)
    targetTemperature:
        object temperature (Kelvin)
    backgroundReflectance:
        background reflectance (unitless)
    backgroundTemperature:
        background temperature (Kelvin)
    haWindspeed:
        the high altitude windspeed (m/s).  Used to calculate the turbulence profile.
    cn2at1m:
        the refractive index structure parameter "near the ground" (e.g. at h = 1 m).
        Used to calculate the turbulence profile.

    :raises: ValueError if ihaze not in acceptable ihaze values
    :raises: ValueError if altitude not in acceptable altitude values
    :raises: ValueError if ground range not in acceptable ground range values
    """

    ihaze_values = [1, 2]
    altitude_values = [2, 32.55, 75, 150, 225, 500] + \
        list(range(1000, 12001, 1000)) + \
        list(range(14000, 20001, 2000)) + [24500]
    groundRange_values = [0, 100, 500] + \
        list(range(1000, 20001, 1000)) + \
        list(range(22000, 80001, 2000)) + \
        list(range(85000, 300001, 5000))

    def __init__(self,
                 name: str,
                 ihaze: int,
                 altitude: float,
                 groundRange: float,

                 aircraftSpeed: Optional[float] = 0.0,
                 targetReflectance: Optional[float] = 0.15,
                 targetTemperature: Optional[float] = 295.0,
                 backgroundReflectance: Optional[float] = 0.07,
                 backgroundTemperature: Optional[float] = 293.0,
                 haWindspeed: Optional[float] = 21.0,
                 cn2at1m: Optional[float] = 1.7e-14
                 ):
        if ihaze not in PybsmScenario.ihaze_values:
            raise ValueError(f"Invalid ihaze value ({ihaze}) must be in {PybsmScenario.ihaze_values}")
        if altitude not in PybsmScenario.altitude_values:
            raise ValueError(f"Invalid altitude value ({altitude})")
        if groundRange not in PybsmScenario.groundRange_values:
            raise ValueError(f"Invalid ground range value ({groundRange})")

        # required parameters
        self.name = name
        self.ihaze = ihaze
        self.altitude = altitude
        self.groundRange = groundRange

        # optional parameters
        self.aircraftSpeed = aircraftSpeed
        self.targetReflectance = targetReflectance
        self.targetTemperature = targetTemperature
        self.backgroundReflectance = backgroundReflectance
        self.backgroundTemperature = backgroundTemperature
        self.haWindspeed = haWindspeed
        self.cn2at1m = cn2at1m

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def create_scenario(self) -> Scenario:
        S = Scenario(self.name, self.ihaze, self.altitude, self.groundRange)
        S.aircraftSpeed = self.aircraftSpeed
        S.targetReflectance = self.targetReflectance
        S.targetTemperature = self.targetTemperature
        S.backgroundReflectance = self.backgroundReflectance
        S.backgroundTemperature = self.backgroundTemperature
        S.haWindspeed = self.haWindspeed
        S.cn2at1m = self.cn2at1m
        return S

    def __call__(
        self
    ) -> Scenario:
        """
        Alias for :meth:`.StoreScenario.create_scenario`.
        """
        return self.create_scenario()

    def get_config(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'ihaze': self.ihaze,
            'altitude': self.altitude,
            'groundRange': self.groundRange,
            'aircraftSpeed': self.aircraftSpeed,
            'targetReflectance': self.targetReflectance,
            'targetTemperature': self.targetTemperature,
            'backgroundReflectance': self.backgroundReflectance,
            'backgroundTemperature': self.backgroundTemperature,
            'haWindspeed': self.haWindspeed,
            'cn2at1m': self.cn2at1m,
        }
