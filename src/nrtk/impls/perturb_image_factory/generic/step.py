import math
from typing import Any, Dict, Optional, Sequence, Type, Union

from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory


class StepPerturbImageFactory(PerturbImageFactory):
    """Simple PerturbImageFactory implementation to step through the given range of values."""

    def __init__(
        self,
        perturber: Type[PerturbImage],
        theta_key: str,
        start: float,
        stop: float,
        step: float = 1.0,
        to_int: Optional[bool] = True,
    ):
        """Initialize the factory to produce PerturbImage instances of the given type.

        Initialize the factory to produce PerturbImage instances of the given type,
        varying the given ``theta_key`` parameter from start to stop with given step.

        :param perturber: Python implementation type of the PerturbImage interface
            to produce.

        :param theta_key: Perturber parameter to vary between instances.

        :param start: Initial value of desired range (inclusive).

        :param stop: Final value of desired range (exclusive).

        :param step: Step value between instances.

        :param to_int: Boolean variable determining whether the thetas are cast as
                       ints or floats.

        :raises TypeError: Given a perturber instance instead of type.
        """
        super().__init__(perturber=perturber, theta_key=theta_key)

        self.to_int = to_int
        self.start = start
        self.stop = stop
        self.step = step

    @property
    def thetas(self) -> Union[Sequence[float], Sequence[int]]:
        if not self.to_int:
            return [self.start + i * self.step for i in range(math.ceil((self.stop - self.start) / self.step))]
        else:
            return [int(self.start + i * self.step) for i in range(math.ceil((self.stop - self.start) / self.step))]

    @property
    def theta_key(self) -> str:
        return super().theta_key

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg["start"] = self.start
        cfg["stop"] = self.stop
        cfg["step"] = self.step
        cfg["to_int"] = self.to_int
        return cfg
