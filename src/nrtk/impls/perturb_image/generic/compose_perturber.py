from typing import Any, Dict, List, Optional, Type, TypeVar

import numpy as np
from smqtk_core.configuration import (
    from_config_dict,
    to_config_dict,
)

from nrtk.interfaces.perturb_image import PerturbImage

C = TypeVar("C", bound="ComposePerturber")


class ComposePerturber(PerturbImage):
    def __init__(self, perturbers: List[PerturbImage]):
        """Initializes the ComposePerturber.

        This has not been tested with perturber factories and is not expected to work wit perturber factories.

        :param perturbers: List of perturbers to apply
        """
        self.perturbers = perturbers

    def perturb(self, image: np.ndarray, additional_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        out_img = image

        if additional_params is None:
            additional_params = dict()

        for perturber in self.perturbers:
            out_img = perturber(out_img, additional_params)

        return out_img

    def get_config(self) -> Dict[str, Any]:
        return {"perturbers": [to_config_dict(perturber) for perturber in self.perturbers]}

    @classmethod
    def from_config(
        cls: Type[C],
        config_dict: Dict,
        merge_default: bool = True,
    ) -> C:
        config_dict = dict(config_dict)

        config_dict["perturbers"] = [
            from_config_dict(perturber, PerturbImage.get_impls()) for perturber in config_dict["perturbers"]
        ]

        return super().from_config(config_dict, merge_default=merge_default)
