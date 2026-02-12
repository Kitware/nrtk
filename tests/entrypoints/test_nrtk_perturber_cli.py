import json
import unittest.mock as mock
from pathlib import Path
from unittest.mock import MagicMock

import py  # type: ignore
import pytest
from click.testing import CliRunner
from maite.protocols.object_detection import Dataset

from nrtk.entrypoints import nrtk_perturber_cli
from tests.interop.maite import DATASET_FOLDER

_STUB_CONFIG = {"PerturberFactory": {"type": "stub"}}


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Write a minimal stub config to a temp file."""
    p = tmp_path / "config.json"
    p.write_text(json.dumps(_STUB_CONFIG))
    return p


@pytest.mark.maite
@pytest.mark.tools
class TestNRTKPerturberCLI:
    """These tests make use of the `tmpdir` fixture from `pytest`.

    Find more information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """

    @mock.patch(
        "nrtk.entrypoints._nrtk_perturber_cli.from_config_dict",
        return_value=MagicMock(__len__=MagicMock(return_value=4)),
    )
    @mock.patch(
        "nrtk.entrypoints._nrtk_perturber_cli.nrtk_perturber",
        return_value=[
            ("_f-0.012_D-0.001_px-2e-05", MagicMock(spec=Dataset)),
            ("_f-0.012_D-0.003_px-2e-05", MagicMock(spec=Dataset)),
            ("_f-0.014_D-0.001_px-2e-05", MagicMock(spec=Dataset)),
            ("_f-0.014_D-0.003_px-2e-05", MagicMock(spec=Dataset)),
        ],
    )
    @mock.patch("nrtk.entrypoints._nrtk_perturber_cli.dataset_to_coco", return_value=None)
    def test_nrtk_perturber(
        self,
        dataset_to_coco_patch: MagicMock,
        entrypoint_patch: MagicMock,
        from_config_dict_patch: MagicMock,  # noqa: ARG002
        config_file: Path,
        tmpdir: py.path.local,
    ) -> None:
        """Test that entrypoint and dataset_to_coco are called with appropriate arguments."""
        output_dir = Path(tmpdir.join(Path("out")))

        runner = CliRunner()
        result = runner.invoke(
            nrtk_perturber_cli,
            [
                f"--dataset_dir={str(DATASET_FOLDER)}",
                f"--output_dir={str(output_dir)}",
                f"--config_file={str(config_file)}",
                "-v",
            ],
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

    @mock.patch("nrtk.entrypoints._nrtk_perturber_cli.from_config_dict")
    @mock.patch("nrtk.entrypoints._nrtk_perturber_cli.nrtk_perturber")
    @mock.patch("pathlib.Path.is_file", side_effect=[True, False])
    def test_missing_metadata_ok(
        self,
        is_file_patch: MagicMock,  # noqa: ARG002
        nrtk_perturber_patch: MagicMock,  # noqa: ARG002
        from_config_dict_patch: MagicMock,  # noqa: ARG002
        config_file: Path,
        caplog: pytest.LogCaptureFixture,
        tmpdir: py.path.local,
    ) -> None:
        """Check that the CLI continues when a metadata file is not present."""
        output_dir = tmpdir.join(Path("out"))

        runner = CliRunner()
        result = runner.invoke(
            nrtk_perturber_cli,
            [
                f"--dataset_dir={str(DATASET_FOLDER)}",
                f"--output_dir={str(output_dir)}",
                f"--config_file={str(config_file)}",
                "-v",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "Could not identify metadata file, assuming no metadata." in caplog.text

    @mock.patch("nrtk.entrypoints._nrtk_perturber_cli.from_config_dict")
    @mock.patch(
        "nrtk.entrypoints._nrtk_perturber_cli.nrtk_perturber",
        side_effect=ValueError("'img_gsd' must be provided for this perturber"),
    )
    @mock.patch("pathlib.Path.is_file", side_effect=[True, False])
    def test_missing_metadata_raises(
        self,
        is_file_patch: MagicMock,  # noqa: ARG002
        nrtk_perturber_patch: MagicMock,  # noqa: ARG002
        from_config_dict_patch: MagicMock,  # noqa: ARG002
        config_file: Path,
        caplog: pytest.LogCaptureFixture,
        tmpdir: py.path.local,
    ) -> None:
        """Check that a ValueError propagates when the perturber requires metadata."""
        output_dir = tmpdir.join(Path("out"))
        runner = CliRunner()

        with pytest.raises(ValueError, match="'img_gsd' must be provided for this perturber"):
            runner.invoke(
                nrtk_perturber_cli,
                [
                    f"--dataset_dir={str(DATASET_FOLDER)}",
                    f"--output_dir={str(output_dir)}",
                    f"--config_file={str(config_file)}",
                    "-v",
                ],
                catch_exceptions=False,
            )

        assert "Could not identify metadata file, assuming no metadata." in caplog.text

    @mock.patch("pathlib.Path.is_file", return_value=False)
    def test_missing_annotations(self, is_file_patch: MagicMock, config_file: Path, tmpdir: py.path.local) -> None:  # noqa: ARG002
        """Check that the CLI exits with code 101 if the annotations file is missing."""
        output_dir = tmpdir.join(Path("out"))

        runner = CliRunner()
        result = runner.invoke(
            nrtk_perturber_cli,
            [
                f"--dataset_dir={str(DATASET_FOLDER)}",
                f"--output_dir={str(output_dir)}",
                f"--config_file={str(config_file)}",
                "-v",
            ],
        )
        assert result.exit_code == 101
