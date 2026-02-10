"""This module contains handle_post, which is the endpoint for NRTK API requests."""

from __future__ import annotations

__all__ = ["handle_post"]

from pathlib import Path

from fastapi import FastAPI, HTTPException

from nrtk.entrypoints import nrtk_perturber
from nrtk.interop._maite.api._converters import build_factory, load_COCOMAITE_dataset
from nrtk.interop._maite.api._dataset_schema import DatasetSchema
from nrtk.interop._maite.api._nrtk_perturb_input_schema import NRTKPerturbInputSchema
from nrtk.interop._maite.api._nrtk_perturb_output_schema import NRTKPerturbOutputSchema
from nrtk.interop._maite.datasets import dataset_to_coco

app = FastAPI()


# Define a route for handling POST requests
@app.post("/")
def handle_post(data: NRTKPerturbInputSchema) -> NRTKPerturbOutputSchema:
    """Returns a collection of augmented datasets based parameters in data.

    Args:
        data:
            NRTKPerturbInputSchema from schema.py

    Returns:
        NRTKPerturbOutputSchema from schema.py

    Raises:
        HTTPException:
            upon failure
    """
    try:
        # Build pybsm factory
        perturber_factory = build_factory(data)

        input_dataset = load_COCOMAITE_dataset(data)

        # Call nrtk_perturber
        augmented_datasets = nrtk_perturber(maite_dataset=input_dataset, perturber_factory=perturber_factory)

        # Format output
        datasets_out = list()
        img_filenames = [Path("images") / img_path.name for img_path in input_dataset.get_img_path_list()]
        for perturb_params, aug_dataset in augmented_datasets:
            full_output_dir = Path(data.output_dir) / perturb_params
            dataset_to_coco(
                dataset=aug_dataset,
                output_dir=full_output_dir,
                img_filenames=img_filenames,
                dataset_categories=input_dataset.get_categories(),
            )
            datasets_out.append(
                DatasetSchema(
                    root_dir=str(full_output_dir),
                    label_file="annotations.json",
                    metadata_file="image_metadata.json",
                ),
            )

        return NRTKPerturbOutputSchema(
            message="Data received successfully",
            datasets=datasets_out,
        )
    except Exception as e:
        # If we made it to this point, we know we do have FastAPI imports
        raise HTTPException(status_code=400, detail=str(e)) from e
