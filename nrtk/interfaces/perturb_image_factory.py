import abc
from typing import Any, Dict, Iterator, Sequence, Type

from smqtk_core import Plugfigurable

from nrtk.interfaces.perturb_image import PerturbImage


class PerturbImageFactory(Plugfigurable):
    """
    Factory class for producing PerturbImage instances of a specified type
    and configuration.
    """

    def __init__(
        self,
        perturber: Type[PerturbImage],
        theta_key: str
    ):
        """
        Initialize the factory to produce PerturbImage instances of the given type,
        varying the given ``theta_key`` parameter.

        :param perturber: Python implementation type of the PerturbImage interface
            to produce.

        :param theta_key: Perturber parameter to vary between instances.

        :raises TypeError: Given a perturber instance instead of type.
        """
        self._theta_key = theta_key

        if not isinstance(perturber, type):  # TODO: this is an incorrect isinstance check
            raise TypeError("Passed a perturber instance, expected type")
        self.perturber = perturber
        self.n = -1

    @property
    @abc.abstractmethod
    def thetas(self) -> Sequence[Any]:
        """ Get the sequence of theta values this factory will iterate over. """

    @property
    def theta_key(self) -> str:
        """ Get the perturber parameter to vary between instances."""
        return self._theta_key

    def __len__(self) -> int:
        """
        :return: Number of perturber instances this factory will generate.
        """
        return len(self.thetas)

    def __iter__(self) -> Iterator[PerturbImage]:
        """
        :return: Iterator for this factory.
        """
        self.n = 0
        return self

    def __next__(self) -> PerturbImage:
        """
        :raises StopIteration: Iterator exhausted.

        :return: Next perturber instance.
        """
        if self.n < len(self.thetas):
            kwargs = {self.theta_key: self.thetas[self.n]}
            func = self.perturber(**kwargs)
            self.n += 1
            return func
        else:
            raise StopIteration

    def __getitem__(self, idx: int) -> PerturbImage:
        """
        Get the perturber for a specific index.

        :param idx: Index of desired perturber.

        :raises IndexError: The given index does not exist.

        :return: Perturber corresponding to the given index.
        """
        if idx < 0 or idx >= len(self.thetas):
            raise IndexError
        kwargs = {self.theta_key: self.thetas[idx]}
        func = self.perturber(**kwargs)
        return func

    def get_config(self) -> Dict[str, Any]:
        return {
            "perturber": self.perturber,
            "theta_key": self.theta_key
        }
