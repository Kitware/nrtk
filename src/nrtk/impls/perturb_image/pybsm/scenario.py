"""
Wrapper for pybsm.scenario.

This module provides a convenient wrapper for setting up and managing scenarios using
the pybsm framework. The primary class, `PybsmScenario`, facilitates configuring scenarios
with parameters such as atmospheric haze, altitude, and ground range.

Typical usage example:

    scenario = PybsmScenario(
        name="example",
        ihaze=1,
        altitude=1000,
        ground_range=500
    )
    out = scenario.create_scenario()

Attributes:
    pybsm_available (bool): Indicates if the pybsm module is available for use.
"""

from typing import Any

try:
    from pybsm.simulation.scenario import Scenario

    pybsm_available = True
except ImportError:
    pybsm_available = False

from smqtk_core import Configurable


class PybsmScenario(Configurable):
    """
    Wrapper for pybsm.scenario.

    This class provides a streamlined interface for creating and configuring a scenario
    within the pybsm framework, enabling the user to specify parameters such as atmospheric
    haze level, altitude, and ground range.

    Attributes:
        ihaze_values (list[int]): Permissible values for the atmospheric haze parameter.
        altitude_values (list[float]): Permissible altitude values for scenario creation.
        ground_range_values (list[float]): Permissible ground range values for the scenario.
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
    ) -> None:
        """
        Initializes a PybsmScenario instance with the specified configuration parameters.

        NOTE:  if the niirs model
        is called, values for target/background temperature, reflectance, etc. are
        overridden with the NIIRS model defaults.

        :parameter ihaze:
            MODTRAN code for visibility, valid options are ihaze = 1 (Rural
            extinction with 23 km visibility) or ihaze = 2 (Rural extinction
            with 5 km visibility)
        :parameter altitude:
            sensor height above ground level in meters; the database includes the
            following altitude options: 2 32.55 75 150 225 500 meters, 1000 to
            12000 in 1000 meter steps, and 14000 to 20000 in 2000 meter steps,
            24500
        :parameter ground_range:
            projection of line of sight between the camera and target along on the
            ground in meters; the distance between the target and the camera is
            given by sqrt(altitude^2 + ground_range^2).
            The following ground ranges are included in the database at each
            altitude until the ground range exceeds the distance to the spherical
            earth horizon: 0 100 500 1000 to 20000 in 1000 meter steps, 22000 to
            80000 in 2000 m steps, and  85000 to 300000 in 5000 meter steps.
        :parameter aircraft_speed:
            ground speed of the aircraft (m/s)
        :parameter target_reflectance:
            object reflectance (unitless); the default 0.15 is the giqe standard
        :parameter target_temperature:
            object temperature (Kelvin); 282 K is used for GIQE calculation
        :parameter background_reflectance:
            background reflectance (unitless)
        :parameter background_temperature:
            background temperature (Kelvin); 280 K used for GIQE calculation
        :parameter ha_wind_speed:
            the high altitude wind speed (m/s) used to calculate the turbulence
            profile; the default, 21.0, is the HV 5/7 profile value
        :parameter cn2_at_1m:
            the refractive index structure parameter "near the ground"
            (e.g. at h = 1 m) used to calculate the turbulence profile; the
            default, 1.7e-14, is the HV 5/7 profile value

        """
        if not self.is_usable():
            raise ImportError("pybsm not found")

        if ihaze not in PybsmScenario.ihaze_values:
            raise ValueError(
                f"Invalid ihaze value ({ihaze}) must be in {PybsmScenario.ihaze_values}",
            )
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

    def create_scenario(self) -> Scenario:
        """
        Creates and returns a pybsm.Scenario object based on the
        provided parameters.

        Returns:
            Scenario: pybsm.Scenario object populated with instance parameters
        """
        S = Scenario(  # noqa:N806
            self.name,
            self.ihaze,
            self.altitude,
            self.ground_range,
        )
        S.aircraft_speed = self.aircraft_speed
        S.target_reflectance = self.target_reflectance
        S.target_temperature = self.target_temperature
        S.background_reflectance = self.background_reflectance
        S.background_temperature = self.background_temperature
        S.ha_wind_speed = self.ha_wind_speed
        S.cn2_at_1m = self.cn2_at_1m
        return S

    def __call__(self) -> "Scenario":
        """Alias for :meth:`.StoreScenario.create_scenario`."""
        return self.create_scenario()

    def get_config(self) -> dict[str, Any]:
        """
        Generates a serializable config that can be used to rehydrate object

        Returns:
            dict[str, Any]: serializable config containing all instance parameters
        """
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

    @classmethod
    def is_usable(cls) -> bool:
        """
        Indicator variable for whether pybsm package is properly installed

        Returns:
            bool: True if pybsm is properly installed, False otherwise
        """
        return pybsm_available
