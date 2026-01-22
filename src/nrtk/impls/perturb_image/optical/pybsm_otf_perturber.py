"""Implements PybsmOTFPerturber which is a base class for the pybsm and otf perturber classes.

Classes:
    PybsmOTFPerturber: Applies OTF-based perturbations to images using pyBSM.

Dependencies:
    - pyBSM for OTF and radiance calculations.
    - nrtk.interfaces.perturb_image.PerturbImage for base functionality.
"""

from __future__ import annotations

__all__ = []

import copy
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.utils._constants import DEFAULT_PYBSM_PARAMS
from nrtk.utils._exceptions import PyBSMImportError
from nrtk.utils._import_guard import import_guard

# Import checks
pybsm_available: bool = import_guard(
    module_name="pybsm",
    exception=PyBSMImportError,
    submodules=["simulation", "simulation.scenario", "simulation.sensor"],
)

from pybsm.simulation import ImageSimulator  # noqa: E402
from pybsm.simulation.scenario import Scenario  # noqa: E402
from pybsm.simulation.sensor import Sensor  # noqa: E402


class PybsmOTFPerturber(PerturbImage, ABC):
    """Base handles common functionality shared across all pybsm-based OTF perturbers.

    This class handles common functionality shared across all pybsm-based OTF perturbers:
    - Sensor/scenario initialization and validation
    - Default parameter handling
    - Image perturbation workflow (GSD extraction, simulation, box rescaling)
    - Configuration management base functionality
    - Dependency checking

    Attributes:
        sensor (Sensor):
            pyBSM sensor object
        scenario (Scenario):
            pyBSM scenario object
        thetas (dict[str, Any]):
            theta parameters related to the perturbation configuration
        interp (bool):
            Specifies whether to use interpolated atmospheric data.
    """

    ihaze_values: list[int] = [1, 2]
    altitude_values: list[float] = (
        [2, 32.55, 75, 150, 225, 500] + list(range(1000, 12001, 1000)) + list(range(14000, 20001, 2000)) + [24500]
    )
    ground_range_values: list[int] = (
        [0, 100, 500]
        + list(range(1000, 20001, 1000))
        + list(range(22000, 80001, 2000))
        + list(range(85000, 300001, 5000))
    )

    def __init__(  # noqa: C901
        self,
        *,
        sensor_name: str = DEFAULT_PYBSM_PARAMS["sensor_name"],
        D: float = DEFAULT_PYBSM_PARAMS["D"],  # noqa:N803
        f: float = DEFAULT_PYBSM_PARAMS["f"],
        p_x: float = DEFAULT_PYBSM_PARAMS["p_x"],
        p_y: float | None = DEFAULT_PYBSM_PARAMS["p_y"],  # Defaults to None since the default value is dependent on p_x
        opt_trans_wavelengths: np.ndarray[Any, Any] = DEFAULT_PYBSM_PARAMS["opt_trans_wavelengths"],
        optics_transmission: np.ndarray[Any, Any] | None = DEFAULT_PYBSM_PARAMS[
            "optics_transmission"
        ],  # Defaults to None since the default value is dependent on opt_trans_wavelengths
        eta: float = DEFAULT_PYBSM_PARAMS["eta"],
        w_x: float | None = DEFAULT_PYBSM_PARAMS["w_x"],  # Defaults to None since the default value is dependent on p_x
        w_y: float | None = DEFAULT_PYBSM_PARAMS["w_y"],  # Defaults to None since the default value is dependent on p_x
        int_time: float = DEFAULT_PYBSM_PARAMS["int_time"],
        n_tdi: float = DEFAULT_PYBSM_PARAMS["n_tdi"],
        dark_current: float = DEFAULT_PYBSM_PARAMS["dark_current"],
        read_noise: float = DEFAULT_PYBSM_PARAMS["read_noise"],
        max_n: int = DEFAULT_PYBSM_PARAMS["max_n"],
        bit_depth: float = DEFAULT_PYBSM_PARAMS["bit_depth"],
        max_well_fill: float = DEFAULT_PYBSM_PARAMS["max_well_fill"],
        s_x: float = DEFAULT_PYBSM_PARAMS["s_x"],
        s_y: float = DEFAULT_PYBSM_PARAMS["s_y"],
        qe_wavelengths: np.ndarray[Any, Any] | None = DEFAULT_PYBSM_PARAMS[
            "qe_wavelengths"
        ],  # Defaults to None since the default value is dependent on opt_trans_wavelengths
        qe: np.ndarray[Any, Any] | None = DEFAULT_PYBSM_PARAMS[
            "qe"
        ],  # Defaults to None since the default value is dependent on opt_trans_wavelengths
        scenario_name: str = DEFAULT_PYBSM_PARAMS["scenario_name"],
        ihaze: int = DEFAULT_PYBSM_PARAMS["ihaze"],
        altitude: float = DEFAULT_PYBSM_PARAMS["altitude"],
        ground_range: float = DEFAULT_PYBSM_PARAMS["ground_range"],
        aircraft_speed: float = DEFAULT_PYBSM_PARAMS["aircraft_speed"],
        target_reflectance: float = DEFAULT_PYBSM_PARAMS["target_reflectance"],
        target_temperature: float = DEFAULT_PYBSM_PARAMS["target_temperature"],
        background_reflectance: float = DEFAULT_PYBSM_PARAMS["background_reflectance"],
        background_temperature: float = DEFAULT_PYBSM_PARAMS["background_temperature"],
        ha_wind_speed: float = DEFAULT_PYBSM_PARAMS["ha_wind_speed"],
        cn2_at_1m: float = DEFAULT_PYBSM_PARAMS["cn2_at_1m"],
        interp: bool = DEFAULT_PYBSM_PARAMS["interp"],
        **kwargs: Any,
    ) -> None:
        """Initialize the pybsm OTF perturber.

        Args:
            sensor_name:
                name of the sensor
            D:
                effective aperture diameter (m)
            f:
                focal length (m)
            p_x:
                detector center-to-center spacings (pitch) in the x and y directions
                (meters); if p_y is not provided, it is assumed equal to p_x
            opt_trans_wavelengths:
                specifies the spectral bandpass of the camera (m); at minimum, specify
                a start and end wavelength
            optics_transmission:
                full system in-band optical transmission (unitless); do not include loss
                due to any telescope obscuration in this optical transmission array
            eta:
                relative linear obscuration (unitless); obscuration of the aperture
                commonly occurs within telescopes due to secondary mirror or spider
                supports
            p_y:
                detector center-to-center spacings (pitch) in the x and y directions
                (meters); if p_y is not provided, it is assumed equal to p_x
            w_x:
                detector width in the x and y directions (m); if set equal to p_x and
                p_y, this corresponds to an assumed full pixel fill factor. In general,
                w_x and w_y are less than p_x and p_y due to non-photo-sensitive area
                (typically transistors) around each pixel.
            w_y:
                detector width in the x and y directions (m); if set equal to p_x and
                p_y, this corresponds to an assumed full pixel fill factor. In general,
                w_x and w_y are less than p_x and p_y due to non-photo-sensitive area
                (typically transistors) around each pixel.
            int_time:
                maximum integration time (s)
            qe:
                quantum efficiency as a function of wavelength (e-/photon)
            qe_wavelengths:
                wavelengths corresponding to the array qe (m)
            dark_current:
                detector dark current (e-/s); dark current is the relatively small
                electric current that flows through photosensitive devices even when no
                photons enter the device
            read_noise:
                amount of noise generated by electronics as the charge present in the pixels
            max_n:
                detector electron well capacity (e-); the default 100 million
                initializes to a large number so that, in the absence of better
                information, it doesn't affect outcomes
            bit_depth:
                resolution of the detector ADC in bits (unitless); default of 100 is a
                sufficiently large number so that in the absence of better information,
                it doesn't affect outcomes
            n_tdi:
                number of TDI stages (unitless)
            max_well_fill:
                maximum amount of charge an individual pixel can hold before it
                becomes saturated
            s_x:
                root-mean-squared jitter amplitudes in the x direction (rad)
            s_y:
                root-mean-squared jitter amplitudes in the y direction (rad)
            scenario_name:
                name of the scenario
            ihaze:
                MODTRAN code for visibility, valid options are ihaze = 1 (Rural
                extinction with 23 km visibility) or ihaze = 2 (Rural extinction
                with 5 km visibility)
            altitude:
                sensor height above ground level in meters; the database includes the
                following altitude options: 2 32.55 75 150 225 500 meters, 1000 to
                12000 in 1000 meter steps, and 14000 to 20000 in 2000 meter steps,
                24500
            ground_range:
                projection of line of sight between the camera and target along on the
                ground in meters; the distance between the target and the camera is
                given by sqrt(altitude^2 + ground_range^2).
                The following ground ranges are included in the database at each
                altitude until the ground range exceeds the distance to the spherical
                earth horizon: 0 100 500 1000 to 20000 in 1000 meter steps, 22000 to
                80000 in 2000 m steps, and  85000 to 300000 in 5000 meter steps.
            aircraft_speed:
                ground speed of the aircraft (m/s)
            target_reflectance:
                object reflectance (unitless); the default 0.15 is the giqe standard
            target_temperature:
                object temperature (Kelvin); 282 K is used for GIQE calculation
            background_reflectance:
                background reflectance (unitless)
            background_temperature:
                background temperature (Kelvin); 280 K used for GIQE calculation
            ha_wind_speed:
                the high altitude wind speed (m/s) used to calculate the turbulence
                profile; the default, 21.0, is the HV 5/7 profile value
            cn2_at_1m:
                the refractive index structure parameter "near the ground"
                (e.g. at h = 1 m) used to calculate the turbulence profile; the
                default, 1.7e-14, is the HV 5/7 profile value
            interp:
                A flag to indicate whether atmospheric interpolation should be used.
                Defaults to False.
            kwargs: sensor and/or scenario values to modify
        Raises:
            ImportError: pyBSM is not found
        """
        if not self.is_usable():
            raise PyBSMImportError
        super().__init__()
        self._simulator: ImageSimulator

        # Convert list inputs to numpy arrays (needed when loading from JSON config)
        opt_trans_wavelengths = np.asarray(opt_trans_wavelengths)
        if optics_transmission is not None:
            optics_transmission = np.asarray(optics_transmission)
        if qe_wavelengths is not None:
            qe_wavelengths = np.asarray(qe_wavelengths)
        if qe is not None:
            qe = np.asarray(qe)

        if ihaze not in PybsmOTFPerturber.ihaze_values:
            raise ValueError(
                f"Invalid ihaze value ({ihaze}) must be in {PybsmOTFPerturber.ihaze_values}",
            )
        if altitude <= PybsmOTFPerturber.altitude_values[-1] and altitude not in PybsmOTFPerturber.altitude_values:
            raise ValueError(f"Invalid altitude value ({altitude})")
        if ground_range not in PybsmOTFPerturber.ground_range_values:
            raise ValueError(f"Invalid ground range value ({ground_range})")

        self._check_opt_trans_wavelengths(opt_trans_wavelengths)

        self.sensor: Sensor = Sensor(
            name=sensor_name,
            D=D,
            f=f,
            p_x=p_x,
            opt_trans_wavelengths=opt_trans_wavelengths,
        )

        self.sensor.optics_transmission = (
            np.ones(opt_trans_wavelengths.shape[0]) if optics_transmission is None else optics_transmission
        )
        self.sensor.eta = eta
        self.sensor.p_y = p_x if p_y is None else p_y
        self.sensor.w_x = p_x if w_x is None else w_x
        self.sensor.w_y = p_x if w_y is None else w_y
        self.sensor.s_x = s_x
        self.sensor.s_y = s_y
        self.sensor.int_time = int_time
        self.sensor.n_tdi = n_tdi
        self.sensor.dark_current = dark_current
        self.sensor.read_noise = read_noise
        self.sensor.max_n = max_n
        self.sensor.max_well_fill = max_well_fill
        self.sensor.bit_depth = bit_depth
        self.sensor.qe_wavelengths = opt_trans_wavelengths if qe_wavelengths is None else qe_wavelengths

        if qe is None:
            _qe = np.ones(opt_trans_wavelengths.shape[0])
        else:
            _qe: np.ndarray = qe
        self.sensor.qe = _qe

        self.scenario: Scenario = Scenario(
            name=scenario_name,
            ihaze=ihaze,
            altitude=altitude,
            ground_range=ground_range,
            interp=interp,
        )
        self.scenario.aircraft_speed = aircraft_speed
        self.scenario.target_reflectance = target_reflectance
        self.scenario.target_temperature = target_temperature
        self.scenario.background_reflectance = background_reflectance
        self.scenario.background_temperature = background_temperature
        self.scenario.ha_wind_speed = ha_wind_speed
        self.scenario.cn2_at_1m = cn2_at_1m

        # Store kwargs for retrieval
        self.thetas: dict[str, Any] = copy.deepcopy(kwargs)
        self._use_default_psf = False

        # self._simulator = self._create_simulator()

    def _check_opt_trans_wavelengths(self, opt_trans_wavelengths: np.ndarray) -> None:
        """Validates the `opt_trans_wavelengths` array to ensure it meets the required criteria.

        This method checks that the `opt_trans_wavelengths` array has at least two elements,
        representing the start and end wavelengths, and that the wavelengths are in ascending order.

        Args:
            opt_trans_wavelengths:
                An array of optical transmission wavelengths.
                The array must contain at least two elements and must be in ascending order.

        Raises:
            ValueError: If `opt_trans_wavelengths` contains fewer than two elements.
            ValueError: If the wavelengths in `opt_trans_wavelengths` are not in ascending order.
        """
        if opt_trans_wavelengths.shape[0] < 2:
            raise ValueError(
                "At minimum, at least the start and end wavelengths must be specified for opt_trans_wavelengths",
            )
        if opt_trans_wavelengths[0] >= opt_trans_wavelengths[-1]:
            raise ValueError("opt_trans_wavelengths must be ascending")

    @abstractmethod
    def _create_simulator(self) -> ImageSimulator:
        """Create the specific ImageSimulator for this perturber."""
        pass

    @override
    def perturb(
        self,
        *,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        img_gsd: float | None = None,
        **additional_params: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Apply the OTF-based perturbation to the provided image.

        Args:
            image:
                The image to be perturbed.
            boxes:
                Bounding boxes for detections in input image.
            img_gsd:
                GSD is the distance between the centers of two adjacent pixels in an image, measured on the ground.
            additional_params:
                Additional perturbation keyword arguments (currently unused).

        Returns:
            The perturbed image and bounding boxes scaled to perturbed image shape.

        Raises:
            ValueError: If 'img_gsd' is None.
        """
        if img_gsd is None:
            raise ValueError("'img_gsd' must be provided for this perturber")

        if self._use_default_psf:
            img_gsd = None

        _, blur_img, noisy_img = self._simulator.simulate_image(image, gsd=img_gsd)

        if self._simulator.add_noise and noisy_img is not None:  # noqa: SIM108
            out_img = noisy_img
        else:
            out_img = blur_img

        # Handle formatting and box rescaling
        return self._handle_boxes_and_format(sim_img=out_img, boxes=boxes, orig_shape=image.shape)

    @override
    def get_config(self) -> dict[str, Any]:
        """Generates a serializable config that can be used to rehydrate object."""
        return {
            "sensor_name": self.sensor_name,
            "D": self.D,
            "f": self.f,
            "p_x": self.p_x,
            "p_y": self.p_y,
            "opt_trans_wavelengths": self.opt_trans_wavelengths.tolist(),
            "optics_transmission": self.optics_transmission.tolist(),
            "eta": self.eta,
            "w_x": self.w_x,
            "w_y": self.w_y,
            "int_time": self.int_time,
            "n_tdi": self.n_tdi,
            "dark_current": self.dark_current,
            "read_noise": self.read_noise,
            "max_n": self.max_n,
            "bit_depth": self.bit_depth,
            "max_well_fill": self.max_well_fill,
            "s_x": self.s_x,
            "s_y": self.s_y,
            "qe_wavelengths": self.qe_wavelengths.tolist(),
            "qe": self.qe.tolist(),
            "scenario_name": self.scenario_name,
            "ihaze": self.ihaze,
            "altitude": self.altitude,
            "ground_range": self.ground_range,
            "aircraft_speed": self.aircraft_speed,
            "target_reflectance": self.target_reflectance,
            "target_temperature": self.target_temperature,
            "background_reflectance": self.background_reflectance,
            "background_temperature": self.background_temperature,
            "ha_wind_speed": self.ha_wind_speed,
            "cn2_at_1m": self.cn2_at_1m,
            "interp": self.interp,
        }

    @classmethod
    def is_usable(cls) -> bool:
        """Check if dependencies are available."""
        return pybsm_available

    @property
    def scenario_name(self) -> str:
        """Getter for scenario_name."""
        return self.scenario.name

    @property
    def ihaze(self) -> float:
        """Getter for ihaze."""
        return self.scenario.ihaze

    @property
    def altitude(self) -> float:
        """Getter for altitude."""
        return self.scenario.altitude

    @property
    def ground_range(self) -> float:
        """Getter for ground_range."""
        return self.scenario.ground_range

    @property
    def target_reflectance(self) -> float:
        """Getter for target_reflectance."""
        return self.scenario.target_reflectance

    @property
    def target_temperature(self) -> float:
        """Getter for target_temperature."""
        return self.scenario.target_temperature

    @property
    def background_reflectance(self) -> float:
        """Getter for background_reflectance."""
        return self.scenario.background_reflectance

    @property
    def background_temperature(self) -> float:
        """Getter for background_temperature."""
        return self.scenario.background_temperature

    @property
    def sensor_name(self) -> str:
        """Getter for sensor_name."""
        return self.sensor.name

    @property
    def p_x(self) -> float:
        """Getter for p_x."""
        return self.sensor.p_x

    @property
    def p_y(self) -> float:
        """Getter for p_y."""
        return self.sensor.p_y

    @property
    def dark_current(self) -> float:
        """Getter for dark_current."""
        return self.sensor.dark_current

    @property
    def read_noise(self) -> float:
        """Getter for read_noise."""
        return self.sensor.read_noise

    @property
    def max_n(self) -> int:
        """Getter for max_n."""
        return self.sensor.max_n

    @property
    def bit_depth(self) -> float:
        """Getter for bit_depth."""
        return self.sensor.bit_depth

    @property
    def max_well_fill(self) -> float:
        """Getter for max_well_fill."""
        return self.sensor.max_well_fill

    @property
    def qe_wavelengths(self) -> NDArray[np.float64]:
        """Getter for qe_wavelengths."""
        return self.sensor.qe_wavelengths

    @property
    def qe(self) -> NDArray[np.float64]:
        """Getter for qe."""
        return self.sensor.qe

    @property
    def opt_trans_wavelengths(self) -> NDArray[np.float64]:
        """Getter for opt_trans_wavelengths."""
        return self.sensor.opt_trans_wavelengths

    @property
    def optics_transmission(self) -> NDArray[np.float64]:
        """Getter for optics_transmission."""
        return self.sensor.optics_transmission

    @property
    def mtf_wavelengths(self) -> NDArray[np.float64]:
        """Getter for mtf_wavelengths."""
        return self._simulator.mtf_wavelengths

    @property
    def mtf_weights(self) -> NDArray[np.float64]:
        """Getter for mtf_weights."""
        return self._simulator.mtf_weights

    @property
    def w_x(self) -> float:
        """Getter for w_x."""
        return self.sensor.w_x

    @property
    def w_y(self) -> float:
        """Getter for w_y."""
        return self.sensor.w_y

    @property
    def s_x(self) -> float:
        """Getter for s_x."""
        return self.sensor.s_x

    @property
    def s_y(self) -> float:
        """Getter for s_y."""
        return self.sensor.s_y

    @property
    def f(self) -> float:
        """Getter for f."""
        return self.sensor.f

    @property
    def D(self) -> float:  # noqa N802
        """Getter for D."""
        return self.sensor.D

    @property
    def eta(self) -> float:
        """Getter for eta."""
        return self.sensor.eta

    @property
    def int_time(self) -> float:
        """Getter for int_time."""
        return self.sensor.int_time

    @property
    def n_tdi(self) -> float:
        """Getter for n_tdi."""
        return self.sensor.n_tdi

    @property
    def slant_range(self) -> float:
        """Getter for slant_range."""
        return self._simulator.slant_range

    @property
    def ha_wind_speed(self) -> float:
        """Getter for ha_wind_speed."""
        return self.scenario.ha_wind_speed

    @property
    def cn2_at_1m(self) -> float:
        """Getter for cn2_at_1m."""
        return self.scenario.cn2_at_1m

    @property
    def aircraft_speed(self) -> float:
        """Getter for aircraft_speed."""
        return self.scenario.aircraft_speed

    @property
    def interp(self) -> bool | None:
        """Getter for interp."""
        return self.scenario._interp  # noqa:SLF001

    @property
    def params(self) -> dict[str, Any]:
        """Retrieves the theta parameters related to the perturbation configuration.

        This method retrieves extra configuration details for the perturber instance,
        which may include specific parameters related to the sensor, scenario, or any
        additional customizations applied during initialization.

        Returns:
            A dictionary containing additional perturbation parameters.
        """
        return self.thetas

    def _handle_boxes_and_format(
        self,
        *,
        sim_img: np.ndarray,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None,
        orig_shape: tuple,
    ) -> tuple[np.ndarray, Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Handle box rescaling and format conversion to uint8."""
        # Convert to uint8
        sim_img_uint8 = np.clip(sim_img, 0, 255).astype(np.uint8)

        # Rescale boxes if provided
        if boxes:
            scaled_boxes = self._rescale_boxes(boxes=boxes, orig_shape=orig_shape, new_shape=sim_img.shape)
            return sim_img_uint8, scaled_boxes

        return sim_img_uint8, boxes
