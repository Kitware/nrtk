"""This module contains nrtk_perturber_cli, which is a CLI script for running nrtk_perturber."""

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

import click  # type: ignore
from smqtk_core.configuration import from_config_dict

from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from nrtk.interop.maite.datasets.object_detection import (
    COCOJATICObjectDetectionDataset,
)
from nrtk.interop.maite.utils.detection import dataset_to_coco
from nrtk.interop.maite.utils.nrtk_perturber import nrtk_perturber
from nrtk.utils._exceptions import KWCocoImportError, MaiteImportError
from nrtk.utils._import_guard import import_guard
from nrtk.utils._logging import setup_logging

kwcoco_available: bool = import_guard(
    module_name="kwcoco",
    exception=KWCocoImportError,
    submodules=["coco_dataset"],
    objects=["CocoDataset"],
)
from kwcoco.coco_dataset import CocoDataset  # noqa: E402

maite_available: bool = import_guard(
    module_name="maite",
    exception=MaiteImportError,
    submodules=["protocols"],
    objects=["DatumMetadata"],
)
from maite.protocols import DatumMetadata  # noqa: E402

logger: logging.Logger = setup_logging(name=__name__)


def _load_metadata(*, dataset_dir: str, kwcoco_dataset: "CocoDataset") -> Sequence[DatumMetadata]:
    metadata_file = Path(dataset_dir) / "image_metadata.json"
    if not metadata_file.is_file():
        logger.warning(
            "Could not identify metadata file, assuming no metadata. Expected at '[dataset_dir]/image_metadata.json'",
        )
        return [DatumMetadata(id=idx) for idx in range(len(kwcoco_dataset.imgs))]
    logger.info(f"Loading metadata from {metadata_file}")
    with open(metadata_file) as f:
        return json.load(f)


def _set_logging(verbose: bool) -> None:
    if verbose:
        logger.setLevel(logging.INFO)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("dataset_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(exists=False))
@click.argument("config_file", type=click.File(mode="r"))
@click.option("--verbose", "-v", count=True, help="print progress messages")
def nrtk_perturber_cli(
    *,
    dataset_dir: str,
    output_dir: str,
    config_file: TextIO,
    verbose: bool,
) -> None:
    """Generate NRTK perturbed images and detections from a given set of source images and COCO-format annotations.

    The perturbed images are stored in subfolders named after the chosen perturbation parameter keys and values.

    Args:
        dataset_dir:
            Root directory of dataset.
        output_dir:
            Directory to write the perturbed images to.
        config_file:
            Configuration file specifying the PerturbImageFactory configuration.
        verbose:
            Display progress messages. Default is false.

    Raises:
        ValueError: COCO annotations file is not found
        KWCocoImportError: KWCOCO is not available
    """
    _set_logging(verbose)

    logger.info(f"Dataset path: {dataset_dir}")

    # Load COCO dataset
    coco_file = Path(dataset_dir) / "annotations.json"
    if not coco_file.is_file():
        raise ValueError("Could not identify annotations file. Expected at '[dataset_dir]/annotations.json'")
    logger.info(f"Loading kwcoco annotations from {coco_file}")
    if not kwcoco_available:
        raise KWCocoImportError
    if not maite_available:
        raise MaiteImportError
    kwcoco_dataset = CocoDataset(data=coco_file)

    # Load metadata, if it exists
    metadata = _load_metadata(dataset_dir=dataset_dir, kwcoco_dataset=kwcoco_dataset)

    # Load config
    config = json.load(config_file)
    perturber_factory = from_config_dict(config=config["PerturberFactory"], type_iter=PerturbImageFactory.get_impls())

    # Initialize dataset object
    input_dataset = COCOJATICObjectDetectionDataset(
        kwcoco_dataset=kwcoco_dataset,
        image_metadata=metadata,
    )

    # Augment input dataset
    augmented_datasets = nrtk_perturber(maite_dataset=input_dataset, perturber_factory=perturber_factory)

    # Save each augmented dataset to its own directory
    output_path = Path(output_dir)
    img_filenames = [Path(img_path.name) for img_path in input_dataset.get_img_path_list()]
    for perturb_params, aug_dataset in augmented_datasets:
        dataset_to_coco(
            dataset=aug_dataset,
            output_dir=output_path / perturb_params,
            img_filenames=img_filenames,
            dataset_categories=input_dataset.get_categories(),
        )
