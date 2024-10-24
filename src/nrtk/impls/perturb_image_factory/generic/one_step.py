from typing import Any, Dict, Type

from nrtk.impls.perturb_image_factory.generic.step import StepPerturbImageFactory
from nrtk.interfaces.perturb_image import PerturbImage


class OneStepPerturbImageFactory(StepPerturbImageFactory):
    """Simple PerturbImageFactory implementation to return a factory with one perturber."""

    def __init__(
        self,
        perturber: Type[PerturbImage],
        theta_key: str,
        theta_value: float,
    ):
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

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg["theta_value"] = self.theta_value
        return {k: cfg[k] for k in ("perturber", "theta_key", "theta_value")}
