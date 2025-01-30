"""This module contains functions to convert input schema to NRTK objects"""

import json
import logging
import os

from smqtk_core.configuration import from_config_dict

from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from nrtk.interop.maite.api.schema import NrtkPerturbInputSchema

try:
    import kwcoco  # type: ignore

    from nrtk.interop.maite.interop.object_detection.dataset import (
        COCOJATICObjectDetectionDataset,
    )

    is_usable = True
except ImportError:
    is_usable = False

LOG = logging.getLogger(__name__)


def build_factory(data: NrtkPerturbInputSchema) -> PerturbImageFactory:
    """Returns a PerturbImageFactory based on scenario and sensor parameters in data.

    :param data: dictionary of Schema from schema.py
    """
    if not os.path.isfile(data.config_file):
        raise FileNotFoundError(f"Config file at {data.config_file} was not found")
    with open(data.config_file) as config_file:
        config = json.load(config_file)
        if "PerturberFactory" not in config:
            raise ValueError(f'Config file at {data.config_file} does not have "PerturberFactory" key')
        return from_config_dict(config["PerturberFactory"], PerturbImageFactory.get_impls())


if not is_usable:
    LOG.warning("load_COCOJATIC_dataset requires additional dependencies, please install 'nrtk-jatic[tools]'")
else:

    def load_COCOJATIC_dataset(  # noqa: N802
        data: NrtkPerturbInputSchema,
    ) -> COCOJATICObjectDetectionDataset:
        """Returns a COCOJATICObjectDetectionDataset based on dataset parameters in data.

        :param data: dictionary of Schema from schema.py
        """
        if not is_usable:
            raise ImportError("This tool requires additional dependencies, please install `nrtk-jatic[tools]`")
        kwcoco_dataset = kwcoco.CocoDataset(data.label_file)

        return COCOJATICObjectDetectionDataset(
            root=data.dataset_dir,
            kwcoco_dataset=kwcoco_dataset,
            image_metadata=data.image_metadata,
        )
