"""
This module provides an implementation of the `ImageMetric` interface, specifically for
calculating the NIIRS (National Imagery Interpretability Rating Scale) metric using pyBSM
sensor and scenario configurations.

Classes:
    NIIRSImageMetric: Computes the NIIRS metric using the provided sensor and scenario.

Dependencies:
    - pybsm.metrics.niirs5
    - nrtk.impls.perturb_image.pybsm.scenario.PybsmScenario
    - nrtk.impls.perturb_image.pybsm.sensor.PybsmSensor

Example usage:
    sensor = PybsmSensor(...)
    scenario = PybsmScenario(...)
    niirs_metric = NIIRSImageMetric(sensor=sensor, scenario=scenario)
    result = niirs_metric.compute_metric(image)
"""

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
    """
    Implementation of the `ImageMetric` interface to calculate the NIIRS metric.

    The NIIRS metric, or National Imagery Interpretability Rating Scale, is used to rate
    the quality of images based on interpretability. This class requires a `PybsmSensor`
    and `PybsmScenario` to be initialized, as it uses these components to perform the metric
    calculation.

    Attributes:
        sensor (PybsmSensor): The sensor configuration for the metric computation.
        scenario (PybsmScenario): The scenario configuration used in metric calculation.
    """

    def __init__(self, sensor: PybsmSensor, scenario: PybsmScenario) -> None:
        """
        Initializes the NIIRSImageMetric.

        Args:
            sensor (PybsmSensor): A pyBSM sensor object representing sensor characteristics.
            scenario (PybsmScenario): A pyBSM scenario object for environmental and context settings.

        Raises:
            ImportError: If the pyBSM library is not available.
            installed via 'nrtk[pybsm-graphics]' or 'nrtk[pybsm-headless]'.
        """
        if not self.is_usable():
            raise ImportError(
                "pybsm not found. Please install 'nrtk[pybsm]', 'nrtk[pybsm-graphics]', or 'nrtk[pybsm-headless]'.",
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
        """
        Given the pyBSMSensor and the pyBSMScenario, compute the NIIRS metric.

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
        """
        Given the pyBSMSensor and the pyBSMScenario, compute the NIIRS metric.

        While this implementation of compute() takes the expected input paramerters, none
        of the values are used during calculation. pyBSM's NIIRS calculation only uses
        the Sensor and Scenario objects to calculate NIIRS and is image independent.

        In order to inherit and function as an ImageMetric implementation, the arguements
        for compute stay consistent with other implementaiotns and are not used.

        :return: Returns the NIIRS metric for the given pyBSMSensor and pyBSMScenario.
        """
        return self.compute()

    @override
    def get_config(self) -> dict[str, Any]:
        """
        Generates a configuration dictionary for the NIIRSImageMetric instance.

        Returns:
            dict[str, Any]: Configuration data representing the sensor and scenario.
        """
        return {
            "sensor": to_config_dict(self.sensor),
            "scenario": to_config_dict(self.scenario),
        }

    @classmethod
    def is_usable(cls) -> bool:
        """
        Checks if the required pybsm module is available.

        Returns:
            bool: True if pybsm is installed; False otherwise.
        """
        return pybsm_available
