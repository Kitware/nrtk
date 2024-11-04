"""
This module provides the `OneStepPerturbImageFactory` class, a simple implementation of the
`StepPerturbImageFactory`. This factory creates a single `PerturbImage` instance with specific
parameters for one-step perturbation, suitable for controlled image transformations.

Classes:
    OneStepPerturbImageFactory: A factory that generates one `PerturbImage` instance
    configured with a specific parameter key and value.

Dependencies:
    - nrtk.impls.perturb_image_factory.generic.step.StepPerturbImageFactory for the base
      factory functionality.
    - nrtk.interfaces.perturb_image.PerturbImage as the interface for the perturber.

Example usage:
    factory = OneStepPerturbImageFactory(perturber=SomePerturbImageClass, theta_key="blur", theta_value=0.5)
    perturbed_image = factory.create()
"""

from typing import Any

from nrtk.impls.perturb_image_factory.generic.step import StepPerturbImageFactory
from nrtk.interfaces.perturb_image import PerturbImage


class OneStepPerturbImageFactory(StepPerturbImageFactory):
    """Simple PerturbImageFactory implementation to return a factory with one perturber."""

    def __init__(
        self,
        perturber: type[PerturbImage],
        theta_key: str,
        theta_value: float,
    ) -> None:
        """Initialize the factory to produce an instance of PerturbImage for the given type.

        Initialize the factory to produce an instance of PerturbImage for the given type,
        given the ``theta_key`` and the ``theta_value`` parameters.

        :param perturber: Python implementation type of the PerturbImage interface
            to produce.

        :param theta_key: Perturber parameter to set for the instance.

        :param theta_value: Initial and only value of ``theta_key``.

        :raises TypeError: Given a perturber instance instead of type.
        """
        super().__init__(perturber=perturber, theta_key=theta_key, start=theta_value, stop=theta_value + 0.1, step=1.0)

        self.theta_value = theta_value

    def get_config(self) -> dict[str, Any]:
        """
        Generates a configuration dictionary for the OneStepPerturbImageFactory instance.

        Returns:
            dict[str, Any]: Configuration data representing the sensor and scenario.
        """
        cfg = super().get_config()
        cfg["theta_value"] = self.theta_value
        return {k: cfg[k] for k in ("perturber", "theta_key", "theta_value")}
