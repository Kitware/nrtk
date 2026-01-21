"""Module for all implementations of PerturbImageFactory."""

from nrtk.impls.perturb_image_factory._perturber_linspace_factory import (
    PerturberLinspaceFactory,
)
from nrtk.impls.perturb_image_factory._perturber_multivariate_factory import (
    PerturberMultivariateFactory,
)
from nrtk.impls.perturb_image_factory._perturber_one_step_factory import (
    PerturberOneStepFactory,
)
from nrtk.impls.perturb_image_factory._perturber_step_factory import (
    PerturberStepFactory,
)

# Override __module__ to reflect the public API path for plugin discovery
PerturberLinspaceFactory.__module__ = __name__
PerturberMultivariateFactory.__module__ = __name__
PerturberOneStepFactory.__module__ = __name__
PerturberStepFactory.__module__ = __name__

__all__ = [
    "PerturberLinspaceFactory",
    "PerturberMultivariateFactory",
    "PerturberOneStepFactory",
    "PerturberStepFactory",
]
