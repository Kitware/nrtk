"""Defines a factory to create PerturbImage instances for flexible image perturbations.

MultivariatePerturbImageFactory: A base factory class that generates multiple `PerturbImage` instances
with specified perturbation parameters.
"""

__all__ = ["MultivariatePerturbImageFactory"]

from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from typing_extensions import override

from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory


class MultivariatePerturbImageFactory(PerturbImageFactory):
    """Base factory for creating `PerturbImage` instances with customizable parameters.

    This factory generates multiple `PerturbImage` instances, each configured with a unique combination
    of specified perturbation parameters (`theta_keys` and `thetas`). These instances allow for flexible
    image perturbation.

    Attributes:
        perturber (type[PerturbImage]): Type of the PerturbImage interface to produce.
        theta_keys (Iterable[str]): Names of parameters to vary across instances.
        _thetas (Sequence[Any]): Values to vary for each parameter in `theta_keys`.
        sets (Sequence[list[int]]): Index combinations for each parameter variation.
    """

    @staticmethod
    def _build_set_list(layer: int, top: Sequence[int]) -> Sequence[list[int]]:
        """Recursively builds a list of index sets to access combinations of parameter values.

        Args:
            layer (int): Current depth of recursion.
            top (Sequence[int]): Maximum index values for each parameter.

        Returns:
            Sequence[list[int]]: A list of index combinations to access parameter values.
        """
        if layer == len(top) - 1:
            return [[i] for i in range(top[layer])]

        return [
            [i] + e for i in range(top[layer]) for e in MultivariatePerturbImageFactory._build_set_list(layer + 1, top)
        ]

    def __init__(
        self,
        perturber: type[PerturbImage],
        theta_keys: Iterable[str],
        thetas: Sequence[Any],
    ) -> None:
        """Initializes the MultivariatePerturbImageFactory.

        :param perturber: type[PerturbImage]: Python implementation type of the PerturbImage interface to produce.
        :param theta_keys: (Sequence[str]): Names of perturbation parameters to vary.
        :param thetas: (Sequence[Any]): Values to use for each perturbation parameter.
        """
        self.perturber = perturber
        self.theta_keys = theta_keys
        self._thetas = thetas

        top = [len(entry) for entry in self.thetas]
        self.sets: Sequence[list[int]] = MultivariatePerturbImageFactory._build_set_list(0, top)
        self.n: int = 0

    def _create_perturber(self, kwargs: dict[str, Any]) -> PerturbImage:
        """Initialize PerturberImage implementation.

        Returns:
            PerturbImage: PerturbImage with specified kwargs
        """
        return self.perturber(**kwargs)

    @override
    def __len__(self) -> int:
        """Returns the number of possible perturbation instances.

        Returns:
            int: The total number of perturbation configurations.
        """
        return len(self.sets)

    @override
    def __iter__(self) -> Iterator[PerturbImage]:
        """Resets the iterator and returns itself for use in for-loops.

        Returns:
            Iterator[PerturbImage]: An iterator over `PerturbImage` instances.
        """
        self.n = 0
        return self

    @override
    def __next__(self) -> PerturbImage:
        """Returns the next `PerturbImage` instance with a unique parameter configuration.

        Returns:
            PerturbImage: A configured `PerturbImage` instance.

        Raises:
            StopIteration: When all configurations have been iterated over.
        """
        if self.n < len(self.sets):
            kwargs = {k: self.thetas[i][self.sets[self.n][i]] for i, k in enumerate(self.theta_keys)}
            func = self._create_perturber(kwargs=kwargs)
            self.n += 1
            return func
        raise StopIteration

    @override
    def __getitem__(self, idx: int) -> PerturbImage:
        """Retrieves a specific `PerturbImage` instance by index.

        Args:
            idx (int): Index of the desired perturbation configuration.

        Returns:
            PerturbImage: The configured `PerturbImage` instance.

        Raises:
            IndexError: If the index is out of range.
        """
        if idx >= len(self.sets):
            raise IndexError("Index out of range")
        kwargs = {k: self.thetas[i][self.sets[idx][i]] for i, k in enumerate(self.theta_keys)}

        return self._create_perturber(kwargs)

    @property
    @override
    def thetas(self) -> Sequence[Sequence[Any]]:
        """Retrieves the current values for each parameter to be varied.

        Returns:
            Sequence[Sequence[Any]]: A sequence of parameter values for perturbation.
        """
        return self._thetas

    @property
    @override
    def theta_key(self) -> str:
        """Returns the parameter key associated with the perturbation settings.

        Returns:
            str: The parameter key name, "params".
        """
        return "params"

    @override
    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration of the `MultivariatePerturbImageFactory` instance.

        Returns:
            dict[str, Any]: Configuration dictionary with current settings.
        """
        return {
            "perturber": self.perturber.get_type_string(),
            "theta_keys": self.theta_keys,
            "thetas": self.thetas,
        }
