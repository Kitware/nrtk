from typing import Any, Dict, Type, Sequence

from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory


class StepPerturbImageFactory(PerturbImageFactory):
    """
    Simple PerturbImageFactory implementation to step through the given range of values.
    """

    def __init__(
        self,
        perturber: Type[PerturbImage],
        theta_key: str,
        start: int,
        stop: int,
        step: int = 1
    ):
        """
        Initialize the factory to produce PerturbImage instances of the given type,
        varying the given ``theta_key`` parameter from start to stop with given step.

        :param perturber: Python implementation type of the PerturbImage interface
            to produce.

        :param theta_key: Perturber parameter to vary between instances.

        :param start: Initial value of desired range (inclusive).

        :param stop: Final value of desired range (exclusive).

        :param step: Step value between instances.

        :raises TypeError: Given a perturber instance instead of type.
        """
        super().__init__(perturber=perturber, theta_key=theta_key)

        self.start = start
        self.stop = stop
        self.step = step

    @property
    def thetas(self) -> Sequence[int]:
        return range(self.start, self.stop, self.step)

    @property
    def theta_key(self) -> str:
        return super().theta_key

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg["start"] = self.start
        cfg["stop"] = self.stop
        cfg["step"] = self.step
        return cfg
