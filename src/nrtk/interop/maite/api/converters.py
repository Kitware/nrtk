"""This module contains functions to convert input schema to NRTK objects"""

import json
import logging
import os

from smqtk_core.configuration import from_config_dict

from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from nrtk.interop.maite.api.schema import NrtkPerturbInputSchema
from nrtk.interop.maite.interop.object_detection.dataset import COCOJATICObjectDetectionDataset
from nrtk.utils._exceptions import KWCocoImportError

try:
    import kwcoco  # type: ignore

    is_usable = True
except ImportError:  # pragma: no cover
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


def load_COCOJATIC_dataset(  # noqa: N802
    data: NrtkPerturbInputSchema,
) -> "COCOJATICObjectDetectionDataset":
    """Returns a COCOJATICObjectDetectionDataset based on dataset parameters in data.

    :param data: dictionary of Schema from schema.py
    """
    if not is_usable:
        raise KWCocoImportError

    for md in data.image_metadata:
        if "id" not in md:
            raise ValueError("ID not present in image metadata. Is it a DatumMetadataType?")

    # PyRight reports that kwcoco and COCOJATICObjectDetectionDataset are possibly unbound due to
    # guarded imports, but we've confirmed they are available with our is_usable check
    kwcoco_dataset = kwcoco.CocoDataset(data.label_file)  # pyright: ignore [reportPossiblyUnboundVariable]
    return COCOJATICObjectDetectionDataset(  # pyright: ignore [reportPossiblyUnboundVariable]
        kwcoco_dataset=kwcoco_dataset,
        # Pydantic doesn't fully support TypedDicts until 3.12+
        # See https://docs.pydantic.dev/2.3/usage/types/dicts_mapping/#typeddict
        # MAITE does not currently import TypedDict via typing_extensions, so we get runtime errors
        #
        # The above ValueError aims to try and error out when the only required key is not present, as that
        # is our only indicator that the metadata is not a DatumMetadataType
        image_metadata=data.image_metadata,  # type: ignore
    )
