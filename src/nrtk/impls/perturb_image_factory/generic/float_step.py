from __future__ import annotations
import numpy as np
from typing import Any, Dict, Type, Sequence

from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory


class FloatStepPerturbImageFactory(PerturbImageFactory):
    """Simple PerturbImageFactory implementation to step through the given range of values."""

    def __init__(
        self,
        perturber: Type[PerturbImage],
        theta_key: str,
        start: float,
        stop: float,
        step: float = 1,
    ):
        """Initialize the factory to produce PerturbImage instances of the given type,
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
    def thetas(self) -> Sequence[float]:
        return np.arange(self.start, self.stop, self.step).tolist()

    @property
    def theta_key(self) -> str:
        return super().theta_key

    def __next__(self) -> PerturbImage:
        """:raises StopIteration: Iterator exhausted.

        :return: Next perturber instance.
        """
        if self.n < len(self.thetas):
            kwargs = {}
            kwargs[self.theta_key] = self.thetas[self.n]
            func = self.perturber(**kwargs)
            self.n += 1
            return func
        else:
            raise StopIteration

    def __getitem__(self, idx: int) -> PerturbImage:
        """Get the perturber for a specific index.

        :param idx: Index of desired perturber.

        :raises IndexError: The given index does not exist.

        :return: Perturber corresponding to the given index.
        """
        if idx < 0 or idx >= len(self.thetas):
            raise IndexError
        kwargs = {}
        kwargs[self.theta_key] = self.thetas[idx]
        func = self.perturber(**kwargs)
        return func

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg["perturber"] = self.perturber.__name__
        cfg["start"] = self.start
        cfg["stop"] = self.stop
        cfg["step"] = self.step
        return cfg

    @classmethod
    def from_config(
        cls: Type[FloatStepPerturbImageFactory], config_dict: Dict, merge_default: bool = True
    ) -> FloatStepPerturbImageFactory:
        config_dict = dict(config_dict)

        perturber_impls = PerturbImage.get_impls()
        type_dict = {
            pert_impl.__name__: pert_impl
            for pert_impl in perturber_impls
        }

        config_dict["perturber"] = type_dict[config_dict["perturber"]]

        return super().from_config(config_dict, merge_default=merge_default)
