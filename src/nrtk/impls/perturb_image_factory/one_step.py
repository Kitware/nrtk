"""Defines OneStepPerturbImageFactory, creating a single PerturbImage with fixed parameters for one-step perturbations.

Classes:
    OneStepPerturbImageFactory: A factory that generates one `PerturbImage` instance
    configured with a specific parameter key and value.

Dependencies:
    - nrtk.impls.perturb_image_factory.step.StepPerturbImageFactory for the base
      factory functionality.
    - nrtk.interfaces.perturb_image.PerturbImage as the interface for the perturber.

Example usage:
    >>> from nrtk.impls.perturb_image.photometric.enhance import BrightnessPerturber
    >>> factory = OneStepPerturbImageFactory(perturber=BrightnessPerturber, theta_key="factor", theta_value=0.5)
"""

__all__ = ["OneStepPerturbImageFactory"]

from typing import Any

from nrtk.impls.perturb_image_factory.step import StepPerturbImageFactory
from nrtk.interfaces.perturb_image import PerturbImage


class OneStepPerturbImageFactory(StepPerturbImageFactory):
    """Simple PerturbImageFactory implementation to return a factory with one perturber.

    Attributes:
        perturber (type[PerturbImage]):
            perturber type to produce
        theta_key (str):
            peturber parameter to modify
        theta_value (float):
            value to set theta_key to
        to_int (bool):
            determines wheter to cast theta_value to a int or float
    """

    def __init__(
        self,
        *,
        perturber: type[PerturbImage],
        theta_key: str,
        theta_value: float,
        to_int: bool = False,
    ) -> None:
        """Initialize the factory to produce an instance of PerturbImage for the given type.

        Initialize the factory to produce an instance of PerturbImage for the given type,
        given the ``theta_key`` and the ``theta_value`` parameters.

        Args:
            perturber:
                Python implementation type of the PerturbImage interface to produce.
            theta_key:
                Perturber parameter to vary between instances.
            theta_value:
                Initial and only value of ``theta_key``.
            to_int:
                Boolean variable determining whether the theta is cast as int or float. Defaults to False.

        Raises:
            TypeError: Given a perturber instance instead of type.
        """
        super().__init__(
            perturber=perturber,
            theta_key=theta_key,
            start=theta_value,
            stop=theta_value + 0.1,
            step=1.0,
            to_int=to_int,
        )

        self.theta_value = theta_value

    def get_config(self) -> dict[str, Any]:
        """Returns a configuration dictionary for the OneStepPerturbImageFactory instance."""
        cfg = super().get_config()
        cfg["theta_value"] = self.theta_value
        return {k: cfg[k] for k in ("perturber", "theta_key", "theta_value", "to_int")}
