"""
This module provides the `PerturbImageFactory` class, an abstract base factory for generating
instances of `PerturbImage` with specified configurations. This factory pattern allows users
to produce various image perturbations by adjusting key parameters in a flexible, reusable way.

Classes:
    PerturbImageFactory: An abstract factory for creating `PerturbImage` instances with specific
    configurations. Allows for custom parameterization of generated instances.

Dependencies:
    - smqtk_core.Plugfigurable for plug-and-play configuration support.
    - nrtk.interfaces.perturb_image.PerturbImage as the base interface for perturbing images.

Example usage:
    factory = PerturbImageFactory(perturber=SomePerturbImageClass, theta_key="altitude")
    for perturber in factory:
        perturber(perturbed_image)
"""

from __future__ import annotations

import abc
from collections.abc import Iterator, Sequence
from typing import Any, TypeVar

from smqtk_core import Plugfigurable

from nrtk.interfaces.perturb_image import PerturbImage

C = TypeVar("C", bound="PerturbImageFactory")


class PerturbImageFactory(Plugfigurable):
    """Factory class for producing PerturbImage instances of a specified type and configuration."""

    def __init__(self, perturber: type[PerturbImage], theta_key: str) -> None:
        """Initialize the factory to produce PerturbImage instances of the given type.

        Initialize the factory to produce PerturbImage instances of the given type,
        varying the given `theta_key` parameter.

        :param perturber: Python implementation type of the PerturbImage interface
            to produce.

        :param theta_key: Perturber parameter to vary between instances.

        :raises TypeError: Given a perturber instance instead of type.
        """
        self._theta_key = theta_key

        if not isinstance(perturber, type):
            raise TypeError("Passed a perturber instance, expected type")
        self.perturber = perturber
        self.n = -1

    @property
    @abc.abstractmethod
    def thetas(self) -> Sequence[Any]:
        """Get the sequence of theta values this factory will iterate over."""

    @property
    def theta_key(self) -> str:
        """Get the perturber parameter to vary between instances."""
        return self._theta_key

    def __len__(self) -> int:
        """:return: Number of perturber instances this factory will generate."""
        return len(self.thetas)

    def __iter__(self) -> Iterator[PerturbImage]:
        """:return: Iterator for this factory."""
        self.n = 0
        return self

    def __next__(self) -> PerturbImage:
        """:raises StopIteration: Iterator exhausted.

        :return: Next perturber instance.
        """
        if self.n < len(self.thetas):
            kwargs = {self.theta_key: self.thetas[self.n]}
            func = self.perturber(**kwargs)
            self.n += 1
            return func
        raise StopIteration

    def __getitem__(self, idx: int) -> PerturbImage:
        """Get the perturber for a specific index.

        :param idx: Index of desired perturber.

        :raises IndexError: The given index does not exist.

        :return: Perturber corresponding to the given index.
        """
        if idx < 0 or idx >= len(self.thetas):
            raise IndexError
        kwargs = {self.theta_key: self.thetas[idx]}

        return self.perturber(**kwargs)

    @classmethod
    def from_config(
        cls: type[C],
        config_dict: dict,
        merge_default: bool = True,
    ) -> C:
        """
        Instantiates a PerturbImageFactory from a configuration dictionary.

        Args:
            config_dict (dict[str, Any]): Configuration dictionary with parameters for instantiation.

        Returns:
            C: An instance of the PerturbImageFactory class.
        """
        config_dict = dict(config_dict)

        # Check to see if there is a perturber key and if it is in bad format
        if "perturber" in config_dict:
            perturber_impls = PerturbImage.get_impls()

            type_dict = {pert_impl.get_type_string(): pert_impl for pert_impl in perturber_impls}

            if config_dict["perturber"] not in type_dict:
                raise ValueError(
                    f"{config_dict['perturber']} is not a valid perturber.",
                )

            config_dict["perturber"] = type_dict[config_dict["perturber"]]

        return super().from_config(config_dict, merge_default=merge_default)

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """
        Returns the default configuration for the PerturbImageFactory.

        This method provides a default configuration dictionary, specifying default
        values for key parameters in the factory. It can be used to create an instance
        of the factory with preset configurations.

        Returns:
            dict[str, Any]: A dictionary containing default configuration parameters.
        """
        cfg = super().get_default_config()
        cfg["perturber"] = PerturbImage.get_type_string()

        return cfg

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the factory instance.

        Returns:
            dict[str, Any]: Configuration dictionary containing the perturber type and theta_key.
        """
        return {
            "perturber": self.perturber.get_type_string(),
            "theta_key": self.theta_key,
        }
