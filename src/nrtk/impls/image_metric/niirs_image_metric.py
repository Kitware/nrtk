import copy
from typing import Any, Dict, Optional

import numpy as np

try:
    from pybsm.metrics import niirs5

    pybsm_available = True
except ImportError:
    pybsm_available = False

from smqtk_core.configuration import to_config_dict

from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
from nrtk.interfaces.image_metric import ImageMetric


class NIIRSImageMetric(ImageMetric):
    """Implementation of the ``ComputeImageMetrics`` interface to calculate the NIIRS metric."""

    def __init__(self, sensor: PybsmSensor, scenario: PybsmScenario) -> None:
        """Initializes the NIIRSImageMetric.

        :param sensor: pyBSM sensor object.
        :param scenario: pyBSM scenario object.

        :raises: ImportError if pyBSM with OpenCV not found,
        installed via 'nrtk[pybsm-graphics]' or 'nrtk[pybsm-headless]'.
        """
        if not self.is_usable():
            raise ImportError(
                "pybsm not found. Please install 'nrtk[pybsm]', 'nrtk[pybsm-graphics]', or 'nrtk[pybsm-headless]'."
            )
        self.sensor = copy.deepcopy(sensor)
        self.scenario = copy.deepcopy(scenario)

    def compute(
        self,
        img_1: Optional[np.ndarray] = None,
        img_2: Optional[np.ndarray] = None,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Given the pyBSMSensor and the pyBSMScenario, compute the NIIRS metric.

        While this implementation of compute() takes the expected input paramerters, none
        of the values are used during calculation. pyBSM's NIIRS calculation only uses
        the Sensor and Scenario objects to calculate NIIRS and is image independent.

        In order to inherit and function as an ImageMetric implementation, the arguements
        for compute stay consistent with other implementaiotns and are not used.

        :return: Returns the NIIRS metric for the given pyBSMSensor and pyBSMScenario.
        """
        metrics = niirs5(self.sensor(), self.scenario())
        return metrics.niirs

    def __call__(
        self,
        img_1: Optional[np.ndarray] = None,
        img_2: Optional[np.ndarray] = None,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calls compute() with the given pyBSMSensor and pyBSMScenario.

        See compute() documentation for more information on input arguements.

        :return: Returns the NIIRS metric for the given pyBSMSensor and pyBSMScenario
        """
        return self.compute()

    def get_config(self) -> Dict[str, Any]:
        config = {
            "sensor": to_config_dict(self.sensor),
            "scenario": to_config_dict(self.scenario),
        }

        return config

    @classmethod
    def is_usable(cls) -> bool:
        return pybsm_available
