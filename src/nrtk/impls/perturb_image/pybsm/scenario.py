from typing import Any, Dict

from pybsm.simulation.scenario import Scenario
from smqtk_core import Configurable


class PybsmScenario(Configurable):
    """Wrapper for pybsm.scenario.

    ihaze:
        MODTRAN code for visibility, valid options are ihaze = 1 (Rural extinction with 23 km visibility)
        or ihaze = 2 (Rural extinction with 5 km visibility)
    altitude:
        sensor height above ground level in meters.  The database includes the following
        altitude options: 2 32.55 75 150 225 500 meters, 1000 to 12000 in 1000 meter steps,
        and 14000 to 20000 in 2000 meter steps, 24500
    ground_range:
        distance *on the ground* between the target and sensor in meters.
        The following ground ranges are included in the database at each altitude
        until the ground range exceeds the distance to the spherical earth horizon:
        0 100 500 1000 to 20000 in 1000 meter steps, 22000 to 80000 in 2000 m steps,
        and  85000 to 300000 in 5000 meter steps.
    aircraft_speed:
        ground speed of the aircraft (m/s)
    target_reflectance:
        object reflectance (unitless)
    target_temperature:
        object temperature (Kelvin)
    background_reflectance:
        background reflectance (unitless)
    background_temperature:
        background temperature (Kelvin)
    ha_wind_speed:
        the high altitude windspeed (m/s).  Used to calculate the turbulence profile.
    cn2_at_1m:
        the refractive index structure parameter "near the ground" (e.g. at h = 1 m).
        Used to calculate the turbulence profile.

    :raises: ValueError if ihaze not in acceptable ihaze values
    :raises: ValueError if altitude not in acceptable altitude values
    :raises: ValueError if ground range not in acceptable ground range values
    """

    ihaze_values = [1, 2]
    altitude_values = (
        [2, 32.55, 75, 150, 225, 500] + list(range(1000, 12001, 1000)) + list(range(14000, 20001, 2000)) + [24500]
    )
    ground_range_values = (
        [0, 100, 500]
        + list(range(1000, 20001, 1000))
        + list(range(22000, 80001, 2000))
        + list(range(85000, 300001, 5000))
    )

    def __init__(
        self,
        name: str,
        ihaze: int,
        altitude: float,
        ground_range: float,
        aircraft_speed: float = 0.0,
        target_reflectance: float = 0.15,
        target_temperature: float = 295.0,
        background_reflectance: float = 0.07,
        background_temperature: float = 293.0,
        ha_wind_speed: float = 21.0,
        cn2_at_1m: float = 1.7e-14,
    ):
        if ihaze not in PybsmScenario.ihaze_values:
            raise ValueError(f"Invalid ihaze value ({ihaze}) must be in {PybsmScenario.ihaze_values}")
        if altitude not in PybsmScenario.altitude_values:
            raise ValueError(f"Invalid altitude value ({altitude})")
        if ground_range not in PybsmScenario.ground_range_values:
            raise ValueError(f"Invalid ground range value ({ground_range})")

        # required parameters
        self.name = name
        self.ihaze = ihaze
        self.altitude = altitude
        self.ground_range = ground_range

        # optional parameters
        self.aircraft_speed = aircraft_speed
        self.target_reflectance = target_reflectance
        self.target_temperature = target_temperature
        self.background_reflectance = background_reflectance
        self.background_temperature = background_temperature
        self.ha_wind_speed = ha_wind_speed
        self.cn2_at_1m = cn2_at_1m

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def create_scenario(self) -> Scenario:
        S = Scenario(self.name, self.ihaze, self.altitude, self.ground_range)  # noqa:N806
        S.aircraft_speed = self.aircraft_speed
        S.target_reflectance = self.target_reflectance
        S.target_temperature = self.target_temperature
        S.background_reflectance = self.background_reflectance
        S.background_temperature = self.background_temperature
        S.ha_wind_speed = self.ha_wind_speed
        S.cn2_at_1m = self.cn2_at_1m
        return S

    def __call__(self) -> Scenario:
        """Alias for :meth:`.StoreScenario.create_scenario`."""
        return self.create_scenario()

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ihaze": self.ihaze,
            "altitude": self.altitude,
            "ground_range": self.ground_range,
            "aircraft_speed": self.aircraft_speed,
            "target_reflectance": self.target_reflectance,
            "target_temperature": self.target_temperature,
            "background_reflectance": self.background_reflectance,
            "background_temperature": self.background_temperature,
            "ha_wind_speed": self.ha_wind_speed,
            "cn2_at_1m": self.cn2_at_1m,
        }
