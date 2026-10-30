"""This module contains nrtk_perturber_cli, which is a CLI script for running nrtk_perturber."""

__all__ = ["nrtk_perturber_cli"]

import json
import logging
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any, TextIO

import click
from datamaite import (
    BoxTrackDataset,
    DatasetFormat,
    DatasetMetadata,
    ObjectDetectionDataset,
    Task,
    VisionDataset,
    available_output_formats,
    load,
    write,
)
from smqtk_core.configuration import from_config_dict

from nrtk.entrypoints._nrtk_eo._nrtk_eo_image_perturber import nrtk_eo_image_perturber
from nrtk.interfaces import PerturbFactory
from nrtk.utils._logging import setup_logging

logger: logging.Logger = setup_logging(name=__name__)


def _set_logging(verbose: bool) -> None:
    if verbose:
        logger.setLevel(logging.INFO)


def _load_metadata(*, dataset_dir: str) -> dict[int | str, dict[str, Any]]:
    metadata_file = Path(dataset_dir) / "datum_metadata.json"
    if not metadata_file.is_file():
        logger.warning(
            "Could not identify metadata file, assuming no metadata. Expected at '[dataset_dir]/datum_metadata.json'",
        )
        return {}
    logger.info(f"Loading metadata from {metadata_file}")
    entries = json.loads(metadata_file.read_text())
    return {entry["id"]: {k: v for k, v in entry.items() if k != "id"} for entry in entries if "id" in entry}


def _combine_object_detection(
    augmented_datasets: list[tuple[str, ObjectDetectionDataset]],
) -> ObjectDetectionDataset:
    """Merge per-perturbation OD datasets into one, re-id-ing images to stay unique."""
    combined_samples = []
    dataset_metadata: DatasetMetadata | None = None
    next_id = 0

    for perturb_params, aug_dataset in augmented_datasets:
        # Carry the dataset metadata through from the first
        # dataset so the merged output stays writable.
        if dataset_metadata is None:
            dataset_metadata = aug_dataset.dataset_metadata
        for sample in aug_dataset.samples:
            combined_samples.append(
                replace(
                    sample,
                    image_id=next_id,
                    file_name=f"images/{next_id}.png",
                    metadata={
                        **sample.metadata,
                        "source_image_id": sample.image_id,
                        "perturber_params": perturb_params,
                    },
                ),
            )
            next_id += 1

    return ObjectDetectionDataset(
        samples=tuple(combined_samples),
        dataset_metadata=dataset_metadata if dataset_metadata is not None else DatasetMetadata(),
        dataset_id="combined",
    )


def _combine_multiobject_tracking(
    augmented_datasets: list[tuple[str, BoxTrackDataset]],
) -> BoxTrackDataset:
    """Merge per-perturbation MOT datasets into one, re-id-ing videos to stay unique."""
    combined_sequences = []
    combined_categories: dict[str, int] = {}
    dataset_metadata: DatasetMetadata | None = None
    next_id = 0

    for perturb_params, aug_dataset in augmented_datasets:
        # Carry the dataset metadata through from the first
        # dataset so the merged output stays writable.
        if dataset_metadata is None:
            dataset_metadata = aug_dataset.dataset_metadata
        combined_categories.update(aug_dataset.categories)
        for sequence in aug_dataset.sequences:
            combined_sequences.append(
                replace(
                    sequence,
                    video_id=next_id,
                    metadata={
                        **sequence.metadata,
                        "source_video_id": sequence.video_id,
                        "perturber_params": perturb_params,
                    },
                ),
            )
            next_id += 1

    return BoxTrackDataset(
        sequences=tuple(combined_sequences),
        categories=combined_categories,
        dataset_metadata=dataset_metadata if dataset_metadata is not None else DatasetMetadata(),
        dataset_id="combined",
    )


def _create_combined_dataset(
    augmented_datasets: Iterable[tuple[str, VisionDataset]],
) -> VisionDataset:
    """Merge the per-perturbation datasets into a single dataset of the same task."""
    datasets = list(augmented_datasets)
    if not datasets:
        raise ValueError("No augmented datasets to combine.")

    first_dataset = datasets[0][1]
    if isinstance(first_dataset, ObjectDetectionDataset):
        return _combine_object_detection(datasets)  # pyright: ignore [reportArgumentType] VisionDataset to ObjectDetectionDataset conversion
    if isinstance(first_dataset, BoxTrackDataset):
        return _combine_multiobject_tracking(datasets)  # pyright: ignore [reportArgumentType] VisionDataset to BoxTrackDataset conversion
    raise TypeError(f"Cannot combine datasets of type {type(first_dataset).__name__}")


def _write_datasets(
    *,
    augmented_datasets: Iterable[tuple[str, VisionDataset]],
    output_dir: str,
    output_dataset_format: DatasetFormat,
    combine_output: bool,
    write_mode: str,
) -> None:
    output_path = Path(output_dir)
    if combine_output:
        combined_dataset = _create_combined_dataset(augmented_datasets=augmented_datasets)
        write(dataset=combined_dataset, dest=output_path, output_format=output_dataset_format, mode=write_mode)
    else:
        for perturb_params, aug_dataset in augmented_datasets:
            write(
                dataset=aug_dataset,
                dest=output_path / perturb_params,
                output_format=output_dataset_format,
                mode=write_mode,
            )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--dataset_dir", "-d", type=click.Path(exists=True), envvar="INPUT_DATASET_PATH")
@click.option("--input_dataset_format", "-i", type=str, default="COCO", envvar="INPUT_DATASET_FORMAT")
@click.option("--output_dir", "-o", type=click.Path(exists=False), envvar="OUTPUT_DATASET_PATH")
@click.option("--output_dataset_format", "-u", type=str, default="COCO", envvar="OUTPUT_DATASET_FORMAT")
@click.option("--config_file", "-c", type=click.File(mode="r"), envvar="CONFIG_FILE")
@click.option("--combine_output", "-m", is_flag=True, envvar="COMBINE_OUTPUT")
@click.option("--enable_experimental", "-e", is_flag=True, envvar="ENABLE_EXPERIMENTAL")
@click.option("--overwrite", "-r", is_flag=True, envvar="OVERWRITE")
@click.option("--verbose", "-v", count=True, help="print progress messages")
def nrtk_perturber_cli(  # noqa: C901 ignore too complex
    *,
    dataset_dir: str,
    input_dataset_format: str,
    output_dir: str,
    output_dataset_format: str,
    config_file: TextIO,
    combine_output: bool,
    enable_experimental: bool,
    overwrite: bool,
    verbose: bool,
) -> None:
    """Generate NRTK perturbed data and detections from a given set of source data.

    The perturbed data are stored in subfolders named after the chosen perturbation parameter keys and values.

    To run the container, use the following command:

    docker run -v /path/to/input:/input/:ro -v /path/to/output:/output/ nrtk-perturber [OPTIONS]

    The /input/ directory mount will contain all files consumed by the entrypoint script and the /output/ directory
    will contain all files produced by the entrypoint script.


    Command Line Options:

    ``--dataset_dir``
        Root directory of dataset.

    ``--input_dataset_format``
        Format of the input dataset. Defaults to COCO.

    ``--output_dir``
        Directory to write the perturbed data to.

    ``--output_dataset_format``
        Format of the output dataset. Defaults to COCO.

    ``--config_file``
        Configuration file specifying the PerturbFactory configuration.

    ``--combine_output``
        If enabled, the output will be one dataset. Defaults to false.

    ``--enable_experimental``
        If enabled, experimental features will be enabled. Defaults to false.

    ``--overwrite``
        If enabled, datamaite will use mode="replace" when writing dataset(s). This will delete
        all files/folders in the output directory. If disabled, datamaite will use mode="error" when
        writing dataset(s). This will throw an error if the output directory is not empty. Defaults to
        false.

    ``--verbose``
        Display progress messages. Default is false.

    If no command line options are given, the entrypoint script will use the following environment variables as inputs:

        ``INPUT_DATASET_PATH``
            Root directory of dataset. Default is ``/input/data/dataset/``.

        ``INPUT_DATASET_FORMAT``
            Format of the input dataset. Default is ``COCO``.

        ``OUTPUT_DATASET_PATH``
            Directory to write out the perturbed data. Default is ``/output/data/result/``.

        ``OUTPUT_DATASET_FORMAT``
            Format of the output dataset. Default is ``COCO``.

        ``CONFIG_FILE``
            Path to JSON configuration file. Default is ``/input/nrtk_config.json``.

        ``COMBINE_OUTPUT``
            Controls if output should be one or many datasets. Default is ``false``.

        ``OVERWRITE``
            Controls if datamaite write mode is "replace" or "error". Default is ``false`` ("error").

        ``ENABLE_EXPERIMENTAL``
            Controls if experimental features should be enabled. Default is ``false``.

    Exits:
        101:
            Error encountered while loading the dataset
        102:
            Input dataset is empty
        103:
            Dataset task not supported
        104:
            Attempting to use experimental features without setting enable_experimental to True
        105:
            Input dataset format is invalid
        106:
            Output dataset format is invalid
        107:
            Output directory is not empty and --overwrite was not given

    """
    _set_logging(verbose)

    logger.info(f"Dataset path: {dataset_dir}")
    logger.info(f"Dataset format: {input_dataset_format}")

    write_mode = "replace" if overwrite else "error"

    try:
        input_format = DatasetFormat(input_dataset_format.lower())
    except ValueError:
        logger.error(f"Input dataset format of {input_dataset_format} is not a valid dataset format")
        sys.exit(105)

    try:
        output_format = DatasetFormat(output_dataset_format.lower())
    except ValueError:
        logger.error(f"Output dataset format of {output_dataset_format} is not a valid dataset format")
        sys.exit(106)

    # Load dataset
    logger.info(f"Loading dataset from {dataset_dir}")
    load_options = {"require_video": True} if input_format is DatasetFormat.HMIE else {}
    dataset: VisionDataset
    try:
        dataset = load(root=dataset_dir, dataset_format=input_format, **load_options)  # pyright: ignore [reportArgumentType] Optional argument for HMIE
    except Exception:
        logger.exception(
            f"Encountered an error while loading dataset at {dataset_dir} with format: {input_dataset_format}.",
        )
        sys.exit(101)
    # Load config
    config = json.load(config_file)

    if len(dataset) == 0:
        logger.error("Input dataset is empty. Some dataset formats (i.e. MOTChallenge) will always be empty.")
        sys.exit(102)

    # Load metadata, if it exists
    metadata = _load_metadata(dataset_dir=dataset_dir)

    valid_output_formats = available_output_formats(task=dataset.task)
    if output_format not in valid_output_formats:
        logger.error(
            f"Output format `{output_format.value}` cannot store a `{dataset.task.value}` dataset; "
            f"valid output formats for this input: {[valid_format.value for valid_format in valid_output_formats]}",
        )
        sys.exit(106)

    # Augment input dataset
    augmented_datasets: Iterable[tuple[str, VisionDataset]]

    # Create a temp dir that will last until writing for video_datasets.
    with tempfile.TemporaryDirectory(prefix="nrtk_eo_video_") as staging_dir:
        if dataset.task == Task.OD:
            perturber_factory = from_config_dict(
                config=config["PerturberFactory"],
                type_iter=PerturbFactory.get_impls(),
            )
            augmented_datasets = nrtk_eo_image_perturber(
                dataset=dataset,  # pyright: ignore [reportArgumentType] VisionDataset to ObjectDetectionDataset conversion
                perturber_factory=perturber_factory,
                metadata=metadata,
            )
        elif dataset.task == Task.MOT:
            if enable_experimental:
                import nrtk.experimental  # noqa: F401
                from nrtk.entrypoints._nrtk_eo._nrtk_eo_video_perturber import nrtk_eo_video_perturber

                perturber_factory = from_config_dict(
                    config=config["PerturberFactory"],
                    type_iter=PerturbFactory.get_impls(),
                )
                augmented_datasets = nrtk_eo_video_perturber(
                    dataset=dataset,  # pyright: ignore [reportArgumentType] VisionDataset to BoxTrackDataset conversion
                    perturber_factory=perturber_factory,
                    metadata=metadata,
                    staging_dir=staging_dir,
                )
            else:
                logger.error(
                    f"Attempting to use experimental features but enable_experimental is {enable_experimental}",
                )
                sys.exit(104)

        else:
            logger.error(f"Dataset task {dataset.task} is not supported")
            sys.exit(103)

        try:
            _write_datasets(
                augmented_datasets=augmented_datasets,
                output_dir=output_dir,
                output_dataset_format=output_format,
                combine_output=combine_output,
                write_mode=write_mode,
            )
        except FileExistsError:
            logger.error(
                f"Output directory {output_dir} is not empty. Pass --overwrite to replace its contents.",
            )
            sys.exit(107)
