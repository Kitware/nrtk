import numpy as np
import pytest

from nrtk.impls.perturb_image.generic.blur import AverageBlurPerturber
from nrtk.impls.perturb_image_factory.generic.step import StepPerturbImageFactory
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from nrtk.interop.maite.datasets.object_detection import (
    JATICDetectionTarget,
    JATICObjectDetectionDataset,
)
from nrtk.interop.maite.utils.nrtk_perturber import nrtk_perturber
from nrtk.utils._import_guard import is_available

deps = ["maite"]
maite_available = [is_available(dep) for dep in deps]


@pytest.mark.skipif(
    not AverageBlurPerturber.is_usable(),
    reason="nrtk.impls.perturb_image.generic.cv2 submodule unavailable.",
)
@pytest.mark.skipif(not maite_available, reason="maite unavailable.")
class TestNRTKPerturberOpenCV:
    """These tests make use of the `tmpdir` fixture from `pytest`.

    Find more information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """

    @pytest.mark.parametrize(
        ("perturber_factory", "img_dirs"),
        [
            (
                StepPerturbImageFactory(
                    perturber=AverageBlurPerturber,
                    theta_key="ksize",
                    start=1,
                    stop=5,
                    step=2,
                    to_int=True,
                ),
                ["_ksize-1", "_ksize-3"],
            ),
        ],
    )
    def test_nrtk_perturber(self, perturber_factory: PerturbImageFactory, img_dirs: list[str]) -> None:
        """Test if the perturber returns the intended number of datasets."""
        num_imgs = 4
        dataset = JATICObjectDetectionDataset(
            imgs=[np.random.default_rng().integers(0, 255, (3, 256, 256), dtype=np.uint8)] * num_imgs,
            dets=[
                JATICDetectionTarget(
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
