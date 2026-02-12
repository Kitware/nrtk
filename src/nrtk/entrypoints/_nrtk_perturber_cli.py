"""This module contains nrtk_perturber_cli, which is a CLI script for running nrtk_perturber."""

__all__ = ["nrtk_perturber_cli"]

import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

import click
from kwcoco.coco_dataset import CocoDataset
from maite.protocols import DatumMetadata
from smqtk_core.configuration import from_config_dict

from nrtk.entrypoints import nrtk_perturber
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from nrtk.interop._maite.datasets import (
    COCOMAITEObjectDetectionDataset,
    dataset_to_coco,
)
from nrtk.utils._logging import setup_logging

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
@click.option("--dataset_dir", "-d", type=click.Path(exists=True), envvar="INPUT_DATASET_PATH")
@click.option("--output_dir", "-o", type=click.Path(exists=False), envvar="OUTPUT_DATASET_PATH")
@click.option("--config_file", "-c", type=click.File(mode="r"), envvar="CONFIG_FILE")
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

    To run the container, use the following command:

    docker run -v /path/to/input:/input/:ro -v /path/to/output:/output/ nrtk-perturber [OPTIONS]

    The /input/ directory mount will contain all files consumed by the entrypoint script and the /output/ directory
    will contain all files produced by the entrypoint script.


    Command Line Options:

        --dataset_dir: Root directory of dataset.

        --output_dir: Directory to write the perturbed images to.

        --config_file: Configuration file specifying the PerturbImageFactory configuration.

        --verbose: Display progress messages. Default is false.

    If no command line options are given, the entrypoint script will use the following environment variables as inputs:

        INPUT_DATASET_PATH: Root directory of dataset. Default is /input/data/dataset/.

        OUTPUT_DATASET_PATH: Directory to write out the perturbed images. Default is /output/data/result/.

        CONFIG_FILE: Path to JSON configuration file. Default is /input/nrtk_config.json.

    Exits:
        101:
            COCO annotations file is not found
    """
    _set_logging(verbose)

    logger.info(f"Dataset path: {dataset_dir}")

    # Load COCO dataset
    coco_file = Path(dataset_dir) / "annotations.json"
    if not coco_file.is_file():
        logger.error("Could not identify annotations file. Expected at '[dataset_dir]/annotations.json'")
        sys.exit(101)

    logger.info(f"Loading kwcoco annotations from {coco_file}")
    kwcoco_dataset = CocoDataset(data=coco_file)

    # Load metadata, if it exists
    metadata = _load_metadata(dataset_dir=dataset_dir, kwcoco_dataset=kwcoco_dataset)

    # Load config
    config = json.load(config_file)
    perturber_factory = from_config_dict(config=config["PerturberFactory"], type_iter=PerturbImageFactory.get_impls())

    # Initialize dataset object
    input_dataset = COCOMAITEObjectDetectionDataset(
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
