"""nrtk_perturber generates augmented MAITE dataset(s) based on a perturber factory configuration."""

__all__ = ["nrtk_perturber"]

import itertools
import logging
from collections.abc import Iterable

import numpy as np

from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from nrtk.interop.maite.interop.object_detection.augmentation import JATICDetectionAugmentation
from nrtk.interop.maite.interop.object_detection.dataset import JATICObjectDetectionDataset
from nrtk.utils._exceptions import MaiteImportError
from nrtk.utils._import_guard import import_guard

maite_available: bool = import_guard("maite", MaiteImportError, ["protocols.object_detection"])
from maite.protocols.object_detection import Dataset  # noqa: E402


def nrtk_perturber(maite_dataset: Dataset, perturber_factory: PerturbImageFactory) -> Iterable[tuple[str, Dataset]]:  # pyright: ignore [reportInvalidTypeForm]
    """Generate augmented dataset(s) of type maite.protocols.object_detection.Dataset.

    Generate augmented dataset(s) of type maite.protocols.object_detection.Dataset
    given an input dataset of the same type and a perturber factory
    implementation. Each perturber dcombination will result in a newly
    generated dataset.

    :param maite_dataset: A dataset object of type maite.protocols.object_detection.Dataset
    :param perturber_factory: PerturbImageFactory implementation.
    """
    if not maite_available:
        raise MaiteImportError

    perturber_factory_config = perturber_factory.get_config()
    if "theta_keys" in perturber_factory_config:  # pyBSM doesn't follow interface rules
        perturb_factory_keys = perturber_factory_config["theta_keys"]
        thetas = perturber_factory.thetas
    else:
        perturb_factory_keys = [perturber_factory.theta_key]
        thetas = [perturber_factory.thetas]

    perturber_combinations = [dict(zip(perturb_factory_keys, v, strict=False)) for v in itertools.product(*thetas)]
    logging.info(f"Perturber sweep values: {perturber_combinations}")

    # Iterate through the different perturber factory parameter combinations and
    # save the perturbed images to disk
    logging.info("Starting perturber sweep")
    augmented_datasets: list[Dataset] = []  # pyright: ignore [reportInvalidTypeForm]
    output_perturb_params: list[str] = []
    for i, (perturber_combo, perturber) in enumerate(zip(perturber_combinations, perturber_factory, strict=False)):
        output_perturb_params.append("".join(f"_{k!s}-{v!s}" for k, v in perturber_combo.items()))

        logging.info(f"Starting perturbation for {output_perturb_params[i]}")

        aug_imgs = []
        aug_dets = []
        aug_metadata = []

        jatic_perturber = JATICDetectionAugmentation(augment=perturber, augment_id=output_perturb_params[i])

        # Formatting data to be of batch_size=1 in order to support MAITE
        # detection protocol's expected input for Augmentation
        for idx in range(len(maite_dataset)):
            aug_img, aug_det, aug_md = jatic_perturber(
                batch=(
                    [maite_dataset[idx][0]],
                    [maite_dataset[idx][1]],
                    [maite_dataset[idx][2]],
                ),
            )
            # Appending data to separate lists in order to handle images
            # of varying sizes
            aug_imgs.append(np.asarray(aug_img)[0])
            aug_dets.append(aug_det[0])
            aug_metadata.append(aug_md[0])

        augmented_datasets.append(
            JATICObjectDetectionDataset(
                imgs=aug_imgs,
                dets=aug_dets,
                datum_metadata=aug_metadata,
                dataset_id=output_perturb_params[i],
            ),
        )
    return zip(output_perturb_params, augmented_datasets, strict=False)
