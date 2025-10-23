"""Implements PybsmPerturber for image perturbations using pyBSM with sensor and scenario configs.

Classes:
    PybsmPerturber: Applies image perturbations using pyBSM based on specified sensor and
    scenario configurations.

Dependencies:
    - pybsm for simulation and reference image functionality.
    - nrtk.impls.perturb_image.pybsm.scenario.PybsmScenario for scenario configuration.
    - nrtk.impls.perturb_image.pybsm.sensor.PybsmSensor for sensor configuration.
    - nrtk.interfaces.perturb_image.PerturbImage as the base interface for image perturbation.

Example usage:
    sensor = PybsmSensor(...)
    scenario = PybsmScenario(...)
    perturber = PybsmPerturber(sensor=sensor, scenario=scenario)
    perturbed_image = perturber.perturb(image)
"""

from __future__ import annotations

__all__ = ["PybsmPerturber"]

import copy
from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from smqtk_core.configuration import (
    from_config_dict,
    make_default_config,
    to_config_dict,
)
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.impls.perturb_image.pybsm.pybsm_otf_perturber import PybsmOTFPerturber
from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
from nrtk.utils._exceptions import PyBSMImportError
from nrtk.utils._import_guard import import_guard

# Import checks
pybsm_available: bool = import_guard("pybsm", PyBSMImportError, ["simulation"])

from pybsm.simulation import ImageSimulator, SystemOTFSimulator  # noqa: E402

DEFAULT_REFLECTANCE_RANGE = np.array([0.05, 0.5])  # It is bad standards to call np.array within argument defaults


class PybsmPerturber(PybsmOTFPerturber):
    """Implements image perturbation using pyBSM sensor and scenario configurations.

    The `PybsmPerturber` class applies realistic perturbations to images by leveraging
    pyBSM's simulation functionalities. It takes in a sensor and scenario, along with
    other optional parameters, to simulate environmental effects on the image.

    See https://pybsm.readthedocs.io/en/latest/explanation.html for image formation concepts and parameter details.

    Attributes:
        sensor (PybsmSensor):
            The sensor configuration for the perturbation.
        scenario (PybsmScenario):
            Scenario settings to apply during the perturbation.
        reflectance_range (np.ndarray):
            Default reflectance range for image simulation.
    """

    def __init__(  # noqa: C901
        self,
        sensor: PybsmSensor | None = None,
        scenario: PybsmScenario | None = None,
        reflectance_range: np.ndarray[Any, Any] = DEFAULT_REFLECTANCE_RANGE,
        rng_seed: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initializes the PybsmPerturber.

        Args:
            sensor:
                pyBSM sensor object.
            scenario:
                pyBSM scenario object.
            reflectance_range:
                Array of reflectances that correspond to pixel values.
            rng_seed:
                integer seed value that will be used for the random number generator.
            kwargs:
                sensor and/or scenario values to modify.

        Raises:
            :raises ImportError: If pyBSM is not found, install via `pip install nrtk[pybsm]`.
            :raises ValueError: If reflectance_range length != 2
            :raises ValueError: If reflectance_range not strictly ascending
        """
        if reflectance_range.shape[0] != 2:
            raise ValueError(f"Reflectance range array must have length of 2, got {reflectance_range.shape[0]}")
        if reflectance_range[0] >= reflectance_range[1]:
            raise ValueError(f"Reflectance range array values must be strictly ascending, got {reflectance_range}")

        # Initialize base class
        super().__init__(sensor=sensor, scenario=scenario)

        # Store perturber-specific overrides
        self._rng = rng_seed
        self._reflectance_range: np.ndarray[Any, Any] = reflectance_range
        for k in kwargs:
            if hasattr(self.sensor, k):
                setattr(self.sensor, k, kwargs[k])
            elif hasattr(self.scenario, k):
                setattr(self.scenario, k, kwargs[k])

        # this is key:value record of the thetas use for perturbing
        self.thetas: dict[str, Any] = copy.deepcopy(kwargs)

        self._simulator = self._create_simulator()

    @override
    def _create_simulator(self) -> ImageSimulator:
        """Create SystemOTFSimulator with explicit parameters."""
        pybsm_sensor = self.sensor.create_sensor()
        pybsm_scenario = self.scenario.create_scenario()
        return SystemOTFSimulator(
            sensor=pybsm_sensor,
            scenario=pybsm_scenario,
            add_noise=True,
            rng=self._rng,
            use_reflectance=True,
            reflectance_range=self._reflectance_range,
        )

    @property
    def params(self) -> dict[str, Any]:
        """Retrieves the theta parameters related to the perturbation configuration.

        This method retrieves extra configuration details for the `PybsmPerturber` instance,
        which may include specific parameters related to the sensor, scenario, or any
        additional customizations applied during initialization.

        Returns:
            :return dict[str, Any]: A dictionary containing additional perturbation parameters.
        """
        return self.thetas

    def __str__(self) -> str:
        """Returns a string representation combining sensor and scenario names.

        Returns:
            :return str: Concatenated sensor and scenario names.
        """
        return self.sensor.name + " " + self.scenario.name

    def __repr__(self) -> str:
        """Returns a representation of the perturber including sensor and scenario names.

        Returns:
            :return str: Representation showing sensor and scenario names.
        """
        return self.__str__()

    @classmethod
    def from_config(cls, config_dict: dict[str, Any], merge_default: bool = True) -> PybsmPerturber:
        """Instantiates a PybsmPerturber from a configuration dictionary.

        Args:
            config_dict:
                Configuration dictionary with initialization parameters.
            merge_default:
                Whether to merge with default configuration. Defaults to True.

        Returns:
            :return PybsmPerturber: An instance of PybsmPerturber.
        """
        config_dict = dict(config_dict)

        config_dict["sensor"] = from_config_dict(config_dict["sensor"], [PybsmSensor])
        config_dict["scenario"] = from_config_dict(config_dict["scenario"], [PybsmScenario])

        # Convert input data to expected constructor types
        config_dict["reflectance_range"] = np.array(config_dict["reflectance_range"])

        return super(PybsmOTFPerturber, cls).from_config(config_dict, merge_default=merge_default)

    @override
    def get_config(self) -> dict[str, Any]:
        """Get current configuration including perturber-specific parameters."""
        cfg = super().get_config()
        cfg["sensor"] = to_config_dict(self._sensor) if self._sensor else None
        cfg["scenario"] = to_config_dict(self._scenario) if self._scenario else None
        cfg["reflectance_range"] = self._reflectance_range.tolist()
        cfg["rng_seed"] = self._rng

        return cfg

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Retrieves the default configuration for PybsmPerturber instances.

        Returns:
            :return dict[str, Any]: A dictionary with the default configuration values.
        """
        cfg = super().get_default_config()
        cfg["sensor"] = make_default_config([PybsmSensor])
        cfg["scenario"] = make_default_config([PybsmScenario])
        cfg["reflectance_range"] = cfg["reflectance_range"].tolist()
        return cfg

    def _handle_boxes_and_format(
        self,
        sim_img: np.ndarray,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None,
        orig_shape: tuple,
    ) -> tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Override to normalize and handle box rescaling and format conversion to uint8."""
        sim = sim_img
        smin, smax = float(sim.min()), float(sim.max())
        if smax > smin:
            sim = (sim - smin) / (smax - smin) * 255.0
        # Convert to uint8
        sim_img_uint8 = sim.astype(np.uint8)

        # Rescale boxes if provided
        if boxes:
            scaled_boxes = self._rescale_boxes(boxes, orig_shape, sim_img.shape)
            return sim_img_uint8, scaled_boxes

        return sim_img_uint8, boxes
