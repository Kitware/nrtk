import unittest.mock as mock
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from unittest.mock import MagicMock

import py  # type: ignore
import pytest
from click.testing import CliRunner

from nrtk.interop.maite.interop.object_detection.utils import maite_available
from nrtk.interop.maite.utils.bin.nrtk_perturber_cli import is_usable, nrtk_perturber_cli
from nrtk.utils._exceptions import KWCocoImportError, MaiteImportError
from tests.interop.maite import DATASET_FOLDER, NRTK_BLUR_CONFIG, NRTK_PYBSM_CONFIG

Dataset: type = object
if maite_available:
    # Multiple type ignores added for pyright's handling of guarded imports
    from maite.protocols.object_detection import Dataset


class TestNRTKPerturberCLI:
    """These tests make use of the `tmpdir` fixture from `pytest`.

    Find more information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """

    @pytest.mark.skipif(not is_usable, reason="Extra 'nrtk-jatic[tools]' not installed.")
    @pytest.mark.skipif(not maite_available, reason=str(MaiteImportError()))
    @mock.patch(
        "nrtk.interop.maite.utils.bin.nrtk_perturber_cli.nrtk_perturber",
        return_value=[
            ("_f-0.012_D-0.001_px-2e-05", MagicMock(spec=Dataset)),  # pyright: ignore [reportPossiblyUnboundVariable]
            ("_f-0.012_D-0.003_px-2e-05", MagicMock(spec=Dataset)),  # pyright: ignore [reportPossiblyUnboundVariable]
            ("_f-0.014_D-0.001_px-2e-05", MagicMock(spec=Dataset)),  # pyright: ignore [reportPossiblyUnboundVariable]
            ("_f-0.014_D-0.003_px-2e-05", MagicMock(spec=Dataset)),  # pyright: ignore [reportPossiblyUnboundVariable]
        ],
    )
    @mock.patch("nrtk.interop.maite.utils.bin.nrtk_perturber_cli.dataset_to_coco", return_value=None)
    def test_nrtk_perturber(
        self,
        dataset_to_coco_patch: MagicMock,
        entrypoint_patch: MagicMock,
        tmpdir: py.path.local,
    ) -> None:
        """Test that entrypoint and dataset_to_coco are called with appropriate arguments."""
        output_dir = Path(tmpdir.join(Path("out")))

        runner = CliRunner()
        result = runner.invoke(
            nrtk_perturber_cli,
            [str(DATASET_FOLDER), str(output_dir), str(NRTK_PYBSM_CONFIG), "-v"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0

        # Confirm entrypoint arguments are as expected
        kwargs = entrypoint_patch.call_args.kwargs
        assert len(kwargs["maite_dataset"]) == 11
        assert kwargs["maite_dataset"]._image_metadata == {idx: {"id": idx, "img_gsd": 0.105} for idx in range(11)}
        assert len(kwargs["perturber_factory"]) == 4

        # Confirm dataset_to_coco arguments are as expected
        img_filenames = [
            Path("0000006_00159_d_0000001.jpg"),
            Path("0000006_00611_d_0000002.jpg"),
            Path("0000006_01111_d_0000003.jpg"),
            Path("0000006_01275_d_0000004.jpg"),
            Path("0000006_01659_d_0000004.jpg"),
            Path("0000006_02138_d_0000006.jpg"),
            Path("0000006_02616_d_0000007.jpg"),
            Path("0000006_03636_d_0000009.jpg"),
            Path("0000006_04050_d_0000010.jpg"),
            Path("0000006_04309_d_0000011.jpg"),
            Path("0000161_01584_d_0000158.jpg"),
        ]
        dset_cats = [
            {"id": 0, "name": "pedestrian", "supercategory": "none"},
            {"id": 1, "name": "people", "supercategory": "none"},
            {"id": 2, "name": "bicycle", "supercategory": "none"},
            {"id": 3, "name": "car", "supercategory": "none"},
            {"id": 4, "name": "van", "supercategory": "none"},
            {"id": 5, "name": "truck", "supercategory": "none"},
            {"id": 6, "name": "tricycle", "supercategory": "none"},
            {"id": 7, "name": "awning-tricycle", "supercategory": "none"},
            {"id": 8, "name": "bus", "supercategory": "none"},
            {"id": 9, "name": "motor", "supercategory": "none"},
        ]
        calls = [
            mock.call(
                dataset=dset,
                output_dir=output_dir / perturb_param,
                img_filenames=img_filenames,
                dataset_categories=dset_cats,
            )
            for perturb_param, dset in entrypoint_patch.return_value
        ]
        dataset_to_coco_patch.assert_has_calls(calls)

    @pytest.mark.skipif(not is_usable, reason="Extra 'nrtk-jatic[tools]' not installed.")
    @pytest.mark.parametrize(
        ("config_file", "expectation"),
        [
            (
                NRTK_PYBSM_CONFIG,
                pytest.raises(
                    ValueError,
                    match="'img_gsd' must be present in image metadata for this perturber",
                ),
            ),
            (NRTK_BLUR_CONFIG, does_not_raise()),
        ],
    )
    @mock.patch("pathlib.Path.is_file", side_effect=[True, False])
    def test_missing_metadata(
        self,
        is_file_patch: MagicMock,  # noqa: ARG002
        config_file: str,
        expectation: AbstractContextManager,
        caplog: pytest.LogCaptureFixture,
        tmpdir: py.path.local,
    ) -> None:
        """Check that the entrypoint is able to continue when a metadata file is not present.

        Check that the entrypoint is able to continue when a metadata file is not present (as long as
        it's not required by the perturber).
        """
        output_dir = tmpdir.join(Path("out"))

        with expectation:
            runner = CliRunner()
            result = runner.invoke(
                nrtk_perturber_cli,
                [str(DATASET_FOLDER), str(output_dir), str(config_file), "-v"],
                catch_exceptions=False,
            )

            assert result.exit_code == 0

        assert "Could not identify metadata file, assuming no metadata." in caplog.text

    @mock.patch("pathlib.Path.is_file", return_value=False)
    def test_missing_annotations(self, tmpdir: py.path.local) -> None:
        """Check that an exception is appropriately raised if the annotations file is missing."""
        output_dir = tmpdir.join(Path("out"))

        with pytest.raises(ValueError, match=r"Could not identify annotations file."):
            CliRunner().invoke(
                nrtk_perturber_cli,
                [str(DATASET_FOLDER), str(output_dir), str(NRTK_PYBSM_CONFIG), "-v"],
                catch_exceptions=False,
            )

    def test_config_gen(self, tmpdir: py.path.local) -> None:
        """Test the generate configuration file option."""
        output_dir = tmpdir.join(Path("out"))

        output_config = tmpdir.join(Path("gen_conf.json"))

        runner = CliRunner()
        runner.invoke(
            nrtk_perturber_cli,
            [
                str(DATASET_FOLDER),
                str(output_dir),
                str(NRTK_PYBSM_CONFIG),
                "-g",
                str(output_config),
            ],
        )

        # Check that config file was created
        assert output_config.check(file=1)
        # Check that no output was generated
        assert not output_dir.check(dir=1)

    @mock.patch("nrtk.interop.maite.utils.bin.nrtk_perturber_cli.is_usable", False)
    def test_missing_deps(self, tmpdir: py.path.local) -> None:
        """Test that proper warning is displayed when required dependencies are not installed."""
        output_dir = tmpdir.join(Path("out"))

        runner = CliRunner()

        with pytest.raises(KWCocoImportError):
            _ = runner.invoke(
                nrtk_perturber_cli,
                [str(DATASET_FOLDER), str(output_dir), str(NRTK_PYBSM_CONFIG)],
                catch_exceptions=False,
            )

        assert not output_dir.check(dir=1)
