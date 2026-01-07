"""Defines a factory to create PerturbImage instances for flexible image perturbations.

PerturberMultivariateFactory: A base factory class that generates multiple `PerturbImage` instances
with specified perturbation parameters.

Dependencies:
    - nrtk.interfaces for the `PerturbImage` and `PerturbImageFactory` interfaces.

Example:
    >>> from nrtk.impls.perturb_image.photometric.enhance import BrightnessPerturber
    >>> factory = PerturberMultivariateFactory(
    ...     perturber=BrightnessPerturber, theta_keys=["factor"], thetas=[[0.1, 0.5]]
    ... )
"""

__all__ = ["PerturberMultivariateFactory"]

from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import numpy as np
from typing_extensions import Self, override

from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory


class PerturberMultivariateFactory(PerturbImageFactory):
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
    def _build_set_list(*, layer: int, top: Sequence[int]) -> Sequence[list[int]]:
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
            [i] + e
            for i in range(top[layer])
            for e in PerturberMultivariateFactory._build_set_list(layer=layer + 1, top=top)
        ]

    def __init__(
        self,
        *,
        perturber: type[PerturbImage],
        theta_keys: Iterable[str],
        thetas: Sequence[Any],
        perturber_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initializes the PerturberMultivariateFactory.

        Args:
            perturber:
                Python implementation type of the PerturbImage interface to produce.
            theta_keys:
                Names of perturbation parameters to vary
            thetas:
                Values to use for each perturbation parameter.
            perturber_kwargs:
                Default kwargs to be used by the perturber. Defaults to {}.
        """
        self.perturber = perturber
        self.theta_keys = theta_keys
        self._thetas = thetas

        top = [len(entry) for entry in self.thetas]
        self.sets: Sequence[list[int]] = PerturberMultivariateFactory._build_set_list(layer=0, top=top)
        self.n: int = 0
        self.perturber_kwargs: dict[str, Any] = {} if perturber_kwargs is None else perturber_kwargs

    def _create_perturber(self, kwargs: dict[str, Any]) -> PerturbImage:
        """Returns PerturberImage implementation with given input args."""
        input_kwargs = self.perturber_kwargs | kwargs
        return self.perturber(**input_kwargs)

    @override
    def __len__(self) -> int:
        """Returns the number of possible perturbation instances."""
        return len(self.sets)

    @override
    def __iter__(self) -> Iterator[PerturbImage]:
        """Resets the iterator and returns itself for use in for-loops."""
        self.n = 0
        return self

    @override
    def __next__(self) -> PerturbImage:
        """Returns the next `PerturbImage` instance with a unique parameter configuration.

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

        return self._create_perturber(kwargs=kwargs)

    @property
    @override
    def thetas(self) -> Sequence[Sequence[Any]]:
        """Returns the current values for each parameter to be varied."""
        return self._thetas

    @property
    @override
    def theta_key(self) -> str:
        """Returns the parameter key associated with the perturbation settings.

        Returns:
            str: The parameter key name, "params".
        """
        return "params"

    @classmethod
    @override
    def from_config(cls, config_dict: dict[str, Any], merge_default: bool = True) -> Self:
        """Rehydrates an object instance from a serializable config dictionary.

        Args:
            cls:
                The class of the object which will be instantiated.
            config_dict:
                Dictionary of serializable values that will be included in the object instance.
            merge_default:
                Indicator variable describing whether or not to use default config values. Defaults to True.

        Returns:
            Instantiation of PerturberMultivariateFactory.
        """
        config_dict = dict(config_dict)

        # Convert input data to expected constructor types
        opt_trans_wavelengths = config_dict["perturber_kwargs"].get("opt_trans_wavelengths", None)
        if opt_trans_wavelengths is not None:
            config_dict["perturber_kwargs"]["opt_trans_wavelengths"] = np.array(
                config_dict["perturber_kwargs"]["opt_trans_wavelengths"],
            )

        # Non-JSON type arguments with defaults (so they might not be there)
        optics_transmission = config_dict["perturber_kwargs"].get("optics_transmission", None)
        if optics_transmission is not None:
            config_dict["perturber_kwargs"]["optics_transmission"] = np.array(
                config_dict["perturber_kwargs"]["optics_transmission"],
            )
        qe_wavelengths = config_dict["perturber_kwargs"].get("qe_wavelengths", None)
        if qe_wavelengths is not None:
            config_dict["perturber_kwargs"]["qe_wavelengths"] = np.array(
                config_dict["perturber_kwargs"]["qe_wavelengths"],
            )
        qe = config_dict["perturber_kwargs"].get("qe", None)
        if qe is not None:
            config_dict["perturber_kwargs"]["qe"] = np.array(config_dict["perturber_kwargs"]["qe"])

        return super().from_config(config_dict, merge_default=merge_default)

    @override
    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration of the `PerturberMultivariateFactory` instance."""
        return {
            "perturber": self.perturber.get_type_string(),
            "theta_keys": self.theta_keys,
            "thetas": self.thetas,
            "perturber_kwargs": self.perturber_kwargs,
        }
