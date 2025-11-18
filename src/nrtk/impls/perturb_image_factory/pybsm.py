"""Defines a factory to create PybsmPerturber instances for flexible image perturbations.

Classes:
    CustomPybsmPerturbImageFactory: A specialized implementation of `MultivariatePerturbImageFactory` with
    preset configurations.

Dependencies:
    - smqtk_core for configuration management.
    - pybsm for pybsm-based perturbation functionalities.
    - nrtk interfaces for image perturbation.

Example usage:
    sensor = PybsmSensor(...)
    scenario = PybsmScenario(...)
    factory = CustomPybsmPerturbImageFactory(sensor=sensor, scenario=scenario,
                                             theta_keys=['key1'], thetas=[[value1, value2]])
    perturber = next(iter(factory))
"""

from __future__ import annotations

__all__ = ["CustomPybsmPerturbImageFactory"]

from collections.abc import Sequence
from typing import Any

from smqtk_core.configuration import (
    from_config_dict,
    make_default_config,
    to_config_dict,
)
from typing_extensions import override

from nrtk.impls.perturb.optical.pybsm_perturber import PybsmPerturber
from nrtk.impls.perturb_image_factory.generic.multivariate import MultivariatePerturbImageFactory
from nrtk.impls.utils.scenario import PybsmScenario
from nrtk.impls.utils.sensor import PybsmSensor
from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.utils._exceptions import PyBSMImportError


class CustomPybsmPerturbImageFactory(MultivariatePerturbImageFactory):
    """A customized version of `MultivariatePerturbImageFactory` with preset configurations.

    This factory extends `MultivariatePerturbImageFactory` to provide a specialized setup for
    creating `PybsmPerturber` instances with predefined sensor, scenario, and parameter values.
    """

    def __init__(
        self,
        sensor: PybsmSensor,
        scenario: PybsmScenario,
        theta_keys: Sequence[str],
        thetas: Sequence[Any],
    ) -> None:
        """Initializes the `CustomPybsmPerturbImageFactory` with sensor, scenario, and parameters.

        See https://pybsm.readthedocs.io/en/latest/explanation.html for image formation concepts and parameter details.

        Args:
            sensor (PybsmSensor): A pyBSM sensor object.
            scenario (PybsmScenario): A pyBSM scenario object.
            theta_keys (Sequence[str]): Names of perturbation parameters to vary.
            thetas (Sequence[Any]): Values to use for each perturbation parameter.
        """
        if not self.is_usable():
            raise PyBSMImportError
        self.sensor = sensor
        self.scenario = scenario
        super().__init__(perturber=PybsmPerturber, theta_keys=theta_keys, thetas=thetas)

    @override
    def _create_perturber(self, kwargs: dict[str, Any]) -> PerturbImage:
        return PybsmPerturber(sensor=self.sensor, scenario=self.scenario, **kwargs)

    @override
    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration of the `CustomPybsmPerturbImageFactory` instance.

        Returns:
            dict[str, Any]: Configuration dictionary with current settings.
        """
        config = super().get_config()
        config.pop("perturber", None)
        config["sensor"] = to_config_dict(self.sensor)
        config["scenario"] = to_config_dict(self.scenario)
        return config

    @override
    @classmethod
    def from_config(
        cls,
        config_dict: dict[str, Any],
        merge_default: bool = True,
    ) -> CustomPybsmPerturbImageFactory:
        """Instantiates a `CustomPybsmPerturbImageFactory` from a configuration dictionary.

        Args:
            config_dict (dict): Configuration dictionary with initialization parameters.
            merge_default (bool, optional): Whether to merge with default configuration. Defaults to True.

        Returns:
            C: An instance of `CustomPybsmPerturbImageFactory`.
        """
        config_dict = dict(config_dict)

        config_dict["sensor"] = from_config_dict(config_dict["sensor"], [PybsmSensor])
        config_dict["scenario"] = from_config_dict(config_dict["scenario"], [PybsmScenario])

        return super().from_config(config_dict, merge_default=merge_default)

    @override
    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Retrieves the default configuration for `CustomPybsmPerturbImageFactory`.

        Returns:
            dict[str, Any]: A dictionary with the default configuration values.
        """
        cfg = super().get_default_config()

        # Remove perturber key if it exists
        cfg.pop("perturber", None)

        cfg["sensor"] = make_default_config([PybsmSensor])
        cfg["scenario"] = make_default_config([PybsmScenario])

        return cfg

    @classmethod
    def is_usable(cls) -> bool:
        """Checks if the required pybsm module is available.

        Returns:
            bool: True if pybsm is installed; False otherwise.
        """
        return all([PybsmPerturber.is_usable(), PybsmScenario.is_usable(), PybsmSensor.is_usable()])
