"""This module contains functions to convert input schema to NRTK objects."""

from __future__ import annotations

__all__ = ["build_factory", "load_COCOMAITE_dataset"]

import json
import os

from kwcoco import CocoDataset
from smqtk_core.configuration import from_config_dict

from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from nrtk.interop._maite.api._nrtk_perturb_input_schema import NRTKPerturbInputSchema
from nrtk.interop._maite.datasets import COCOMAITEObjectDetectionDataset


def build_factory(data: NRTKPerturbInputSchema) -> PerturbImageFactory:
    """Returns a PerturbImageFactory based on scenario and sensor parameters in data.

    Args:
        data:
            dictionary of Schema from schema.py

    Raises:
        FileNotFoundError:
            if data.config_file does not exists
        ValueError:
            if data.config_file does not have PerturberFactory key
    """
    if not os.path.isfile(data.config_file):
        raise FileNotFoundError(f"Config file at {data.config_file} was not found")
    with open(data.config_file) as config_file:
        config = json.load(config_file)
        if "PerturberFactory" not in config:
            raise ValueError(f'Config file at {data.config_file} does not have "PerturberFactory" key')
        return from_config_dict(config=config["PerturberFactory"], type_iter=PerturbImageFactory.get_impls())


def load_COCOMAITE_dataset(  # noqa: N802
    data: NRTKPerturbInputSchema,
) -> COCOMAITEObjectDetectionDataset:
    """Returns a COCOMAITEObjectDetectionDataset based on dataset parameters in data.

    Args:
        data:
            dictionary of Schema from schema.py

    Raises:
        ValueError:
            data.image_metadata does not have "id" key
    """
    for md in data.image_metadata:
        if "id" not in md:
            raise ValueError("ID not present in image metadata. Is it a DatumMetadataType?")

    # PyRight reports that kwcoco and COCOMAITEObjectDetectionDataset are possibly unbound due to
    # guarded imports, but at this point the module has been imported successfully so they are available
    kwcoco_dataset = CocoDataset(data.label_file)  # pyright: ignore [reportCallIssue]
    return COCOMAITEObjectDetectionDataset(
        kwcoco_dataset=kwcoco_dataset,
        # Pydantic doesn't fully support TypedDicts until 3.12+
        # See https://docs.pydantic.dev/2.3/usage/types/dicts_mapping/#typeddict
        # MAITE does not currently import TypedDict via typing_extensions, so we get runtime errors
        #
        # The above ValueError aims to try and error out when the only required key is not present, as that
        # is our only indicator that the metadata is not a DatumMetadataType
        image_metadata=data.image_metadata,  # type: ignore
    )
