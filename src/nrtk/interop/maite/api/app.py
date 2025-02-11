"""This module contains handle_post, which is the endpoint for NRTK API requests"""

from pathlib import Path
from typing import Callable, Union

try:
    from fastapi import FastAPI, HTTPException

    fastapi_available = True
except ImportError:  # pragma: no cover
    fastapi_available = False

from nrtk.interop.maite.api.converters import build_factory, load_COCOJATIC_dataset
from nrtk.interop.maite.api.schema import (
    DatasetSchema,
    NrtkPerturbInputSchema,
    NrtkPerturbOutputSchema,
)
from nrtk.interop.maite.interop.object_detection.utils import dataset_to_coco
from nrtk.interop.maite.utils.nrtk_perturber import nrtk_perturber
from nrtk.utils._exceptions import FastApiImportError


class _UnusableFastApi:
    def post(self, _: str) -> Callable:
        def wrapper(_: str) -> None:
            return None

        return wrapper


if fastapi_available:
    app: Union[FastAPI, _UnusableFastApi] = FastAPI()  # pyright: ignore [reportPossiblyUnboundVariable]
else:
    app = _UnusableFastApi()


# Define a route for handling POST requests
@app.post("/")
def handle_post(data: NrtkPerturbInputSchema) -> NrtkPerturbOutputSchema:
    """Returns a collection of augmented datasets based parameters in data.

    :param data: NrtkPybsmPerturbInputSchema from schema.py

    :returns: NrtkPybsmPerturberOutputSchema from schema.py

    :raises: HTTPException upon failure
    """
    if not fastapi_available:
        raise HTTPException(status_code=400, detail=str(FastApiImportError()))

    try:
        # Build pybsm factory
        perturber_factory = build_factory(data)

        input_dataset = load_COCOJATIC_dataset(data)

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

        return NrtkPerturbOutputSchema(
            message="Data received successfully",
            datasets=datasets_out,
        )
    except Exception as e:
        # If we made it to this point, we know we do have FastAPI imports
        raise HTTPException(status_code=400, detail=str(e)) from e  # pyright: ignore [reportPossiblyUnboundVariable]
