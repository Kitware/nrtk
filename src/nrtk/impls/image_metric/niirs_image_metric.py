from __future__ import annotations

import copy
from typing import Any

import numpy as np

try:
    from pybsm.metrics import niirs5

    pybsm_available = True
except ImportError:
    pybsm_available = False

from smqtk_core.configuration import to_config_dict
from typing_extensions import override

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

    @override
    def compute(
        self,
        img_1: np.ndarray | None = None,
        img_2: np.ndarray | None = None,
        additional_params: dict[str, Any] | None = None,
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

    @override
    def __call__(
        self,
        img_1: np.ndarray | None = None,
        img_2: np.ndarray | None = None,
        additional_params: dict[str, Any] | None = None,
    ) -> float:
        return self.compute()

    @override
    def get_config(self) -> dict[str, Any]:
        return {
            "sensor": to_config_dict(self.sensor),
            "scenario": to_config_dict(self.scenario),
        }

        return config

    @classmethod
    def is_usable(cls) -> bool:
        return pybsm_available
