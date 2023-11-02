import copy
from typing import Any, Dict

import numpy as np
import pybsm

from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.interfaces.perturb_image import PerturbImage


class PybsmPerturber(PerturbImage):
    def __init__(
            self,
            sensor: PybsmSensor,
            scenario: PybsmScenario,
            reflectance_range: np.ndarray = np.array([.05, .5]),
            **kwargs: Any
            ) -> None:

        """
        :param sensor: pyBSM sensor object.
        :param scenario: pyBSM scenario object.
        :param reflectance_range: Array of reflectances that correspond to pixel values.
        """
        self.sensor = copy.deepcopy(sensor)
        self.scenario = copy.deepcopy(scenario)

        for k in kwargs:
            if hasattr(self.sensor, k):
                setattr(self.sensor, k, kwargs[k])
            elif hasattr(self.scenario, k):
                setattr(self.scenario, k, kwargs[k])

        self.metrics = pybsm.niirs(self.sensor, self.scenario)

        assert reflectance_range.shape[0] == 2
        assert reflectance_range[0] < reflectance_range[1]
        self.reflectance_range = reflectance_range

        # this is key:value record of the thetas use for perturbing
        self.thetas = copy.deepcopy(kwargs)

    @property
    def params(self) -> Dict:
        return self.thetas

    def perturb(
            self,
            image: np.ndarray,
            additional_params: Dict[str, Any] = {}
            ) -> np.ndarray:
        assert 'img_gsd' in additional_params

        perturbed = pybsm.metrics2image(
                    self.metrics,
                    image,
                    additional_params['img_gsd'],
                    np.array([image.min(), image.max()]),
                    self.reflectance_range)[-1]

        min = perturbed.min()
        den = perturbed.max()-min
        perturbed -= min
        perturbed /= den
        perturbed *= 255

        return perturbed.astype(np.uint8)

    def __call__(
            self,
            image: np.ndarray,
            additional_params: Dict[str, Any] = {}
            ) -> np.ndarray:
        """
        Alias for :meth:`.NIIRS.apply`.
        """
        return self.perturb(image, additional_params)

    def __str__(self) -> str:
        return self.metrics.name

    def __repr__(self) -> str:
        return self.metrics.name

    def get_config(self) -> Dict[str, Any]:
        config = {
            'sensor': self.sensor.get_config(),
            'scenario': self.scenario.get_config(),
            'reflectance_range': self.reflectance_range
        }

        for k in self.sensor.__dict__:
            config[k] = getattr(self.sensor, k)
        for k in self.scenario.__dict__:
            config[k] = getattr(self.scenario, k)

        return config
