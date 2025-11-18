import json
from pathlib import Path

import numpy as np
import pytest

from nrtk.impls.perturb_image_factory.pybsm import CustomPybsmPerturbImageFactory
from nrtk.impls.utils.scenario import PybsmScenario
from nrtk.impls.utils.sensor import PybsmSensor
from nrtk.interop.maite.datasets.object_detection import COCOJATICObjectDetectionDataset
from nrtk.interop.maite.utils.detection import maite_available
from nrtk.interop.maite.utils.nrtk_perturber import nrtk_perturber
from nrtk.utils._exceptions import KWCocoImportError, MaiteImportError
from nrtk.utils._import_guard import import_guard
from tests.interop.maite import DATASET_FOLDER

kwcoco_available: bool = import_guard("kwcoco", KWCocoImportError)
import kwcoco  # type: ignore  # noqa: E402
from maite.protocols.object_detection import DatumMetadataType  # noqa: E402


def _load_dataset(dataset_path: str, load_metadata: bool = True) -> COCOJATICObjectDetectionDataset:
    coco_file = Path(dataset_path) / "annotations.json"
    # Type ignore added for pyright's handling of guarded imports
    kwcoco_dataset = kwcoco.CocoDataset(coco_file)  # pyright: ignore [reportCallIssue]

    if load_metadata:
        metadata_file = Path(dataset_path) / "image_metadata.json"
        with open(metadata_file) as f:
            metadata: list[DatumMetadataType] = json.load(f)  # pyright: ignore [reportInvalidTypeForm]
    else:
        metadata: list[DatumMetadataType] = [{"id": idx} for idx in range(len(kwcoco_dataset.imgs))]  # pyright: ignore [reportInvalidTypeForm]

    # Initialize dataset object
    return COCOJATICObjectDetectionDataset(
        kwcoco_dataset=kwcoco_dataset,
        image_metadata=metadata,
    )


@pytest.mark.skipif(
    not COCOJATICObjectDetectionDataset.is_usable(),
    reason="COCOJATICObjectDetectionDataset not usable",
)
@pytest.mark.skipif(not CustomPybsmPerturbImageFactory.is_usable(), reason="CustomPybsmPerturbImageFactory not usable")
@pytest.mark.skipif(not PybsmSensor.is_usable(), reason="PybsmSensor not usable")
@pytest.mark.skipif(not PybsmScenario.is_usable(), reason="PybsmScenario not usable")
@pytest.mark.skipif(not kwcoco_available, reason=str(KWCocoImportError()))
@pytest.mark.skipif(not maite_available, reason=str(MaiteImportError()))
class TestNRTKPerturberPyBSM:
    """These tests make use of the `tmpdir` fixture from `pytest`.

    Find more information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """

    def test_missing_metadata(self) -> None:
        """Test that an appropriate error is raised if required metadata is missing."""
        dataset = _load_dataset(dataset_path=str(DATASET_FOLDER), load_metadata=False)

        pybsm_factory = CustomPybsmPerturbImageFactory(
            sensor=PybsmSensor(
                name="L32511x",
                D=0.004,
                f=0.014285714285714287,
                p_x=0.00002,
                opt_trans_wavelengths=np.array([3.8e-7, 7.0e-7]),
                eta=0.4,
                int_time=0.3,
                read_noise=25.0,
                max_n=96000,
                bit_depth=11.9,
                max_well_fill=0.005,
                da_x=0.0001,
                da_y=0.0001,
                qe_wavelengths=np.array([3.0e-7, 4.0e-7, 5.0e-7, 6.0e-7, 7.0e-7, 8.0e-7, 9.0e-7, 1.0e-6, 1.1e-6]),
                qe=np.array([0.05, 0.6, 0.75, 0.85, 0.85, 0.75, 0.5, 0.2, 0]),
            ),
            scenario=PybsmScenario(name="niceday", ihaze=2, altitude=75, ground_range=0),
            theta_keys=["f", "D"],
            thetas=[[0.014, 0.012], [0.001]],
        )

        with pytest.raises(
            ValueError,
            match="'img_gsd' must be provided for this perturber",
        ):
            _ = nrtk_perturber(maite_dataset=dataset, perturber_factory=pybsm_factory)
