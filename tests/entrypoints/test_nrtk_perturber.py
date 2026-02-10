import numpy as np
import pytest

from nrtk.entrypoints import nrtk_perturber
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from nrtk.interop._maite.datasets import (
    MAITEObjectDetectionDataset,
    MAITEObjectDetectionTarget,
)
from tests.fakes import FakePerturber, PerturberFakeFactory


@pytest.mark.maite
class TestNRTKPerturber:
    """These tests make use of the `tmpdir` fixture from `pytest`.

    Find more information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """

    @pytest.mark.parametrize(
        ("perturber_factory", "img_dirs"),
        [
            (
                PerturberFakeFactory(
                    perturber=FakePerturber,
                    theta_key="param1",
                    theta_values=[1, 3],
                ),
                ["_param1-1", "_param1-3"],
            ),
        ],
    )
    def test_nrtk_perturber(self, perturber_factory: PerturbImageFactory, img_dirs: list[str]) -> None:
        """Test if the perturber returns the intended number of datasets."""
        num_imgs = 4
        dataset = MAITEObjectDetectionDataset(
            imgs=[np.random.default_rng().integers(low=0, high=255, size=(3, 256, 256), dtype=np.uint8)] * num_imgs,
            dets=[
                MAITEObjectDetectionTarget(
                    boxes=np.array([[1.0, 2.0, 3.0, 4.0]]),
                    labels=np.array([0]),
                    scores=np.array([0.5]),
                ),
            ]
            * num_imgs,
            datum_metadata=[{"id": idx} for idx in range(num_imgs)],
            dataset_id="test_dataset",
        )

        augmented_datasets = nrtk_perturber(maite_dataset=dataset, perturber_factory=perturber_factory)

        for perturber_params, aug_dataset in augmented_datasets:
            assert perturber_params in list(img_dirs)
            assert len(aug_dataset) == num_imgs
