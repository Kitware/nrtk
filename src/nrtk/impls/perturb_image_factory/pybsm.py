from __future__ import annotations

from importlib.util import find_spec
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Type, TypeVar

from smqtk_core.configuration import (
    from_config_dict,
    make_default_config,
    to_config_dict,
)

from nrtk.impls.perturb_image.pybsm.perturber import PybsmPerturber
from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory

C = TypeVar("C", bound="_PybsmPerturbImageFactory")


class _PybsmPerturbImageFactory(PerturbImageFactory):
    @staticmethod
    def _build_set_list(layer: int, top: Sequence[int]) -> Sequence[List[int]]:
        if layer == len(top) - 1:
            return [[i] for i in range(top[layer])]
        out = []
        for i in range(top[layer]):
            rec = _PybsmPerturbImageFactory._build_set_list(layer + 1, top)
            for e in rec:
                out.append([i] + e)
        return out

    def __init__(
        self,
        sensor: PybsmSensor,
        scenario: PybsmScenario,
        theta_keys: Iterable[str],
        thetas: Sequence[Any],
    ) -> None:
        """Initializes the PybsmPerturbImageFactory.

        :param sensor: pyBSM sensor object.
        :param scenario: pyBSM scenario object.
        :param theta_keys: Perturber parameter(s) to vary between instances.
        :param theta_keys: Perturber parameter(s) values to vary between instances.
        """
        if not self.is_usable():
            raise ImportError("OpenCV not found. Please install 'nrtk[graphics]' or 'nrtk[headless]'.")
        self.sensor = sensor
        self.scenario = scenario
        self.theta_keys = theta_keys
        self._thetas = thetas

        top = [len(entry) for entry in self.thetas]
        self.sets = _PybsmPerturbImageFactory._build_set_list(0, top)

    def __len__(self) -> int:
        return len(self.sets)

    def __iter__(self) -> Iterator[PerturbImage]:
        self.n = 0
        return self

    def __next__(self) -> PerturbImage:
        if self.n < len(self.sets):
            kwargs = {k: self.thetas[i][self.sets[self.n][i]] for i, k in enumerate(self.theta_keys)}
            func = PybsmPerturber(self.sensor, self.scenario, **kwargs)
            self.n += 1
            return func
        else:
            raise StopIteration

    def __getitem__(self, idx: int) -> PerturbImage:
        if idx >= len(self.sets):
            raise IndexError("Index out of range")
        kwargs = {k: self.thetas[i][self.sets[idx][i]] for i, k in enumerate(self.theta_keys)}
        func = PybsmPerturber(self.sensor, self.scenario, **kwargs)
        return func

    @property
    def thetas(self) -> Sequence[Sequence[Any]]:
        return self._thetas

    @property
    def theta_key(self) -> str:
        return "params"

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()

        # Remove perturber key if it exists
        cfg.pop("perturber", None)

        cfg["sensor"] = make_default_config([PybsmSensor])
        cfg["scenario"] = make_default_config([PybsmScenario])

        return cfg

    @classmethod
    def from_config(
        cls: Type[C],
        config_dict: Dict,
        merge_default: bool = True,
    ) -> C:
        config_dict = dict(config_dict)

        config_dict["sensor"] = from_config_dict(config_dict["sensor"], [PybsmSensor])
        config_dict["scenario"] = from_config_dict(config_dict["scenario"], [PybsmScenario])

        return super().from_config(config_dict, merge_default=merge_default)

    def get_config(self) -> Dict[str, Any]:
        return {
            "theta_keys": self.theta_keys,
            "sensor": to_config_dict(self.sensor),
            "scenario": to_config_dict(self.scenario),
            "thetas": self.thetas,
        }

    @classmethod
    def is_usable(cls) -> bool:
        # Requires pyBSM which requires opencv to be installed
        return find_spec("cv2") is not None


class CustomPybsmPerturbImageFactory(_PybsmPerturbImageFactory):
    def __init__(
        self,
        sensor: PybsmSensor,
        scenario: PybsmScenario,
        theta_keys: Sequence[str],
        thetas: Sequence[Any],
    ):
        super().__init__(sensor, scenario, theta_keys, thetas)
