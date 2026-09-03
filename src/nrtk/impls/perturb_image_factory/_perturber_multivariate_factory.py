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

from typing_extensions import deprecated, override

from nrtk.impls.perturb_factory._perturber_multivariate_factory import (
    PerturberMultivariateFactory as GenericMultivariateFactory,
)
from nrtk.interfaces import PerturbImage, PerturbImageFactory


@deprecated("Use nrtk.impls.perturb_factory.PerturberMultivariateFactory instead.")
class PerturberMultivariateFactory(PerturbImageFactory):
    """Deprecated base factory for creating `PerturbImage` instances with customizable parameters.

    This factory generates multiple `PerturbImage` instances, each configured with a unique combination
    of specified perturbation parameters (`theta_keys` and `thetas`). These instances allow for flexible
    image perturbation.

    .. deprecated:: 1.1
        Use :class:`nrtk.impls.perturb_factory.PerturberMultivariateFactory` instead.
        :mod:`nrtk.impls.perturb_image_factory` will be removed in a future major release.

    Attributes:
        perturber (type[PerturbT_co]):
            Type of the perturber to produce.

            .. deprecated:: 1.1
                Use get_config() instead.
        theta_keys (Iterable[str]):
            Names of parameters to vary across instances.

            .. deprecated:: 1.1
                Use get_config() instead.
        sets (Sequence[list[int]]):
            Index combinations for each parameter variation.

            .. deprecated:: 1.1
                This property will be removed in a future major release.
    """

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

        Raises:
            TypeError:
                If perturber is an instance instead of a type.
            ValueError:
                If theta_keys is empty or theta_keys and thetas have different lengths.
        """
        self._new_impl = GenericMultivariateFactory(
            perturber=perturber,
            theta_keys=theta_keys,
            thetas=thetas,
            perturber_kwargs=perturber_kwargs,
        )
        self._iterator = iter(self._new_impl)

    @override
    def _create_perturber(self, kwargs: dict[str, Any]) -> PerturbImage:
        """Returns perturber implementation with given input args."""
        return self._new_impl._create_perturber(kwargs)  # noqa: SLF001

    @override
    def __len__(self) -> int:
        """Returns the number of possible perturbation instances."""
        return len(self._new_impl)

    @override
    def __iter__(self) -> Iterator[PerturbImage]:
        """Resets the iterator and returns itself for use in for-loops."""
        self._iterator = iter(self._new_impl)
        return self

    @override
    def __next__(self) -> PerturbImage:
        """Returns the next `PerturbImage` instance with a unique parameter configuration.

        Raises:
            StopIteration:
                When all configurations have been iterated over.
        """
        return next(self._iterator)

    @override
    def __getitem__(self, idx: int) -> PerturbImage:
        """Retrieves a specific `PerturbImage` instance by index.

        Args:
            idx: Index of the desired perturbation configuration (supports negative indices).

        Returns:
            PerturbImage: The configured `PerturbImage` instance.
        """
        return self._new_impl[idx]

    @property
    @override
    def thetas(self) -> Sequence[Sequence[Any]]:
        """Returns the current values for each parameter to be varied."""
        return self._new_impl.thetas

    @property
    @override
    @deprecated(
        "This property will be removed in a future major release.",
    )
    def theta_key(self) -> str:
        """Returns the parameter key associated with the perturbation settings.

        Returns:
            str: The parameter key name, "params".
        """
        return self._new_impl.theta_key

    @property
    @deprecated(
        "Use get_config() instead.",
    )
    def theta_keys(self) -> Iterable[str]:
        return self._new_impl.theta_keys

    @theta_keys.setter
    @deprecated(
        "Setting this property will be removed in a future major release.",
    )
    def theta_keys(self, theta_keys: Iterable[str]) -> None:
        self._new_impl.theta_keys = theta_keys

    @property
    @deprecated(
        "Use get_config() instead.",
    )
    @override
    def perturber(self) -> type[PerturbImage]:
        return self._new_impl.perturber

    @perturber.setter
    @deprecated(
        "Setting this property will be removed in a future major release.",
    )
    def perturber(self, perturber: type[PerturbImage]) -> None:
        self._new_impl.perturber = perturber

    @property
    @deprecated(
        "This property will be removed in a future major release.",
    )
    def sets(self) -> Sequence[list[int]]:
        return self._new_impl.sets

    @sets.setter
    @deprecated(
        "This property will be removed in a future major release.",
    )
    def sets(self, sets: Sequence[list[int]]) -> None:
        self._new_impl.sets = sets

    @override
    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration of the `PerturberMultivariateFactory` instance."""
        return self._new_impl.get_config()
