import itertools
import json
import shutil
import unittest.mock as mock
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import py  # type: ignore
import pytest
from click.testing import CliRunner
from datamaite import DatasetFormat, Task
from PIL import Image
from syrupy.assertion import SnapshotAssertion
from syrupy.extensions.json import JSONSnapshotExtension

from nrtk.entrypoints import nrtk_perturber_cli
from nrtk.entrypoints._nrtk_perturber_cli import _load_metadata
from tests.entrypoints import DATUM_METADATA
from tests.impls import INPUT_DRONE_VIDEO_FILE_PATH
from tests.interop.maite import DATASET_FOLDER, NRTK_BRIGHTNESS_CONFIG
from tests.utils.video_io import read_video

_STUB_CONFIG = {"PerturberFactory": {"type": "stub"}}

# Mock target prefix for symbols imported into the CLI module namespace.
_CLI = "nrtk.entrypoints._nrtk_perturber_cli"


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Write a minimal stub config to a temp file (from_config_dict is mocked)."""
    p = tmp_path / "config.json"
    p.write_text(json.dumps(_STUB_CONFIG))
    return p


def _fake_dataset(*, task: Task, length: int = 4) -> MagicMock:
    """A stand-in for a loaded datamaite dataset: has a ``task`` and a length."""
    dataset = MagicMock()
    dataset.__len__.return_value = length
    dataset.task = task
    return dataset


def _augmented() -> list[tuple[str, MagicMock]]:
    """Fake ``(perturb_params, dataset)`` output of the perturber entrypoints."""
    return [
        ("_f-0.012_D-0.001", MagicMock()),
        ("_f-0.014_D-0.003", MagicMock()),
    ]


def _write_refusing_nonempty(*, mode: str, **_: Any) -> None:
    """Stand-in for ``datamaite.write`` against a non-empty destination."""
    if mode == "error":
        raise FileExistsError("Output directory is not empty")


@pytest.mark.maite
@pytest.mark.tools
class TestNRTKPerturberCLI:
    """These tests make use of the `tmpdir` fixture from `pytest`.

    Find more information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """

    @mock.patch(f"{_CLI}.write")
    @mock.patch(
        "nrtk.entrypoints._nrtk_eo._nrtk_eo_video_perturber.nrtk_eo_video_perturber",
    )  # Cannot use _CLI since FMV is a conditional import
    @mock.patch(f"{_CLI}.nrtk_eo_image_perturber")
    @mock.patch(f"{_CLI}.from_config_dict")
    @mock.patch(f"{_CLI}.load")
    def test_object_detection_path(
        self,
        load_patch: MagicMock,
        from_config_dict_patch: MagicMock,
        image_perturber_patch: MagicMock,
        video_perturber_patch: MagicMock,
        write_patch: MagicMock,
        config_file: Path,
        tmpdir: py.path.local,
    ) -> None:
        """An OD dataset dispatches to the image perturber and writes one dataset per subdir."""
        load_patch.return_value = _fake_dataset(task=Task.OD)
        perturber_factory = from_config_dict_patch.return_value
        augmented = _augmented()
        image_perturber_patch.return_value = augmented

        output_dir = Path(tmpdir.join(Path("out")))
        result = CliRunner().invoke(
            cli=nrtk_perturber_cli,
            args=[
                f"--dataset_dir={tmpdir}",
                "--input_dataset_format=COCO",
                f"--output_dir={output_dir}",
                "--output_dataset_format=COCO",
                f"--config_file={config_file}",
                "-v",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        load_patch.assert_called_once_with(root=str(tmpdir), dataset_format=DatasetFormat.COCO)

        image_perturber_patch.assert_called_once_with(
            dataset=load_patch.return_value,
            perturber_factory=perturber_factory,
            metadata={},
        )
        video_perturber_patch.assert_not_called()

        assert write_patch.call_count == len(augmented)
        for perturb_params, aug_dataset in augmented:
            write_patch.assert_any_call(
                dataset=aug_dataset,
                dest=output_dir / perturb_params,
                output_format=DatasetFormat.COCO,
                mode="error",
            )

    @mock.patch(f"{_CLI}.tempfile.TemporaryDirectory")
    @mock.patch(f"{_CLI}.write")
    @mock.patch(
        "nrtk.entrypoints._nrtk_eo._nrtk_eo_video_perturber.nrtk_eo_video_perturber",
    )  # Cannot use _CLI since FMV is a conditional import
    @mock.patch(f"{_CLI}.nrtk_eo_image_perturber")
    @mock.patch(f"{_CLI}.from_config_dict")
    @mock.patch(f"{_CLI}.load")
    def test_multiobject_tracking_path(
        self,
        load_patch: MagicMock,
        from_config_dict_patch: MagicMock,
        image_perturber_patch: MagicMock,
        video_perturber_patch: MagicMock,
        write_patch: MagicMock,
        tempdir_patch: MagicMock,
        config_file: Path,
        tmpdir: py.path.local,
    ) -> None:
        """A MOT dataset dispatches to the video perturber instead of the image one."""
        load_patch.return_value = _fake_dataset(task=Task.MOT)
        perturber_factory = from_config_dict_patch.return_value
        augmented = _augmented()
        video_perturber_patch.return_value = augmented
        tempdir_patch.return_value.__enter__.return_value = "/fake/staging"

        output_dir = Path(tmpdir.join(Path("out")))
        result = CliRunner().invoke(
            cli=nrtk_perturber_cli,
            args=[
                f"--dataset_dir={tmpdir}",
                "--input_dataset_format=hmie",
                f"--output_dir={output_dir}",
                "--output_dataset_format=hmie",
                f"--config_file={config_file}",
                "--enable_experimental",
                "-v",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        video_perturber_patch.assert_called_once_with(
            dataset=load_patch.return_value,
            perturber_factory=perturber_factory,
            metadata={},
            staging_dir="/fake/staging",
        )
        image_perturber_patch.assert_not_called()

        assert write_patch.call_count == len(augmented)
        for perturb_params, aug_dataset in augmented:
            write_patch.assert_any_call(
                dataset=aug_dataset,
                dest=output_dir / perturb_params,
                output_format=DatasetFormat.HMIE,
                mode="error",
            )

    @mock.patch(f"{_CLI}.write")
    @mock.patch(f"{_CLI}._create_combined_dataset")
    @mock.patch(f"{_CLI}.nrtk_eo_image_perturber")
    @mock.patch(f"{_CLI}.from_config_dict")
    @mock.patch(f"{_CLI}.load")
    def test_combine_output(
        self,
        load_patch: MagicMock,
        from_config_dict_patch: MagicMock,  # noqa: ARG002
        image_perturber_patch: MagicMock,
        create_combined_patch: MagicMock,
        write_patch: MagicMock,
        config_file: Path,
        tmpdir: py.path.local,
    ) -> None:
        """--combine_output merges the datasets and writes a single output."""
        load_patch.return_value = _fake_dataset(task=Task.OD)
        augmented = _augmented()
        image_perturber_patch.return_value = augmented

        output_dir = Path(tmpdir.join(Path("out")))
        result = CliRunner().invoke(
            cli=nrtk_perturber_cli,
            args=[
                f"--dataset_dir={tmpdir}",
                "--input_dataset_format=COCO",
                f"--output_dir={output_dir}",
                "--output_dataset_format=COCO",
                f"--config_file={config_file}",
                "--combine_output",
                "-v",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        # The augmented datasets are merged into one, then written a single time.
        create_combined_patch.assert_called_once_with(augmented_datasets=augmented)
        write_patch.assert_called_once_with(
            dataset=create_combined_patch.return_value,
            dest=output_dir,
            output_format=DatasetFormat.COCO,
            mode="error",
        )

    @mock.patch(f"{_CLI}.write")
    @mock.patch(f"{_CLI}.nrtk_eo_image_perturber")
    @mock.patch(f"{_CLI}.from_config_dict")
    @mock.patch(f"{_CLI}.load")
    def test_write_without_overwrite(
        self,
        load_patch: MagicMock,
        from_config_dict_patch: MagicMock,  # noqa: ARG002
        image_perturber_patch: MagicMock,
        write_patch: MagicMock,
        config_file: Path,
        tmpdir: py.path.local,
    ) -> None:
        """Test lack of --overwrite causes exit code 107."""
        load_patch.return_value = _fake_dataset(task=Task.OD)
        augmented = _augmented()
        image_perturber_patch.return_value = augmented
        write_patch.side_effect = _write_refusing_nonempty

        output_dir = Path(tmpdir.join(Path("out")))
        result = CliRunner().invoke(
            cli=nrtk_perturber_cli,
            args=[
                f"--dataset_dir={tmpdir}",
                "--input_dataset_format=COCO",
                f"--output_dir={output_dir}",
                "--output_dataset_format=COCO",
                f"--config_file={config_file}",
            ],
        )
        assert result.exit_code == 107

    @mock.patch(f"{_CLI}.write")
    @mock.patch(f"{_CLI}.nrtk_eo_image_perturber")
    @mock.patch(f"{_CLI}.from_config_dict")
    @mock.patch(f"{_CLI}.load")
    def test_write_with_overwrite(
        self,
        load_patch: MagicMock,
        from_config_dict_patch: MagicMock,  # noqa: ARG002
        image_perturber_patch: MagicMock,
        write_patch: MagicMock,
        config_file: Path,
        tmpdir: py.path.local,
    ) -> None:
        """Test --overwrite causes no issues writing."""
        load_patch.return_value = _fake_dataset(task=Task.OD)
        augmented = _augmented()
        image_perturber_patch.return_value = augmented
        write_patch.side_effect = _write_refusing_nonempty

        output_dir = Path(tmpdir.join(Path("out")))
        result = CliRunner().invoke(
            cli=nrtk_perturber_cli,
            args=[
                f"--dataset_dir={tmpdir}",
                "--input_dataset_format=COCO",
                f"--output_dir={output_dir}",
                "--output_dataset_format=COCO",
                f"--config_file={config_file}",
                "--overwrite",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        assert write_patch.call_count == len(augmented)
        for perturb_params, aug_dataset in augmented:
            write_patch.assert_any_call(
                dataset=aug_dataset,
                dest=output_dir / perturb_params,
                output_format=DatasetFormat.COCO,
                mode="replace",
            )

    @mock.patch(f"{_CLI}.from_config_dict")
    @mock.patch(f"{_CLI}.load")
    def test_load_failure(
        self,
        load_patch: MagicMock,
        from_config_dict_patch: MagicMock,
        config_file: Path,
        tmpdir: py.path.local,
    ) -> None:
        """Check that the CLI exits with code 101 when the dataset fails to load."""
        load_patch.side_effect = RuntimeError("corrupt dataset")

        output_dir = Path(tmpdir.join(Path("out")))
        result = CliRunner().invoke(
            cli=nrtk_perturber_cli,
            args=[
                f"--dataset_dir={tmpdir}",
                "--input_dataset_format=COCO",
                f"--output_dir={output_dir}",
                "--output_dataset_format=COCO",
                f"--config_file={config_file}",
                "-v",
            ],
        )
        assert result.exit_code == 101
        from_config_dict_patch.assert_not_called()

    @mock.patch(f"{_CLI}.from_config_dict")
    @mock.patch(f"{_CLI}.load")
    def test_empty_dataset(
        self,
        load_patch: MagicMock,
        from_config_dict_patch: MagicMock,  # noqa: ARG002
        config_file: Path,
        tmpdir: py.path.local,
    ) -> None:
        """Check that the CLI exits with code 102 if the input dataset is empty."""
        load_patch.return_value = _fake_dataset(task=Task.OD, length=0)

        output_dir = Path(tmpdir.join(Path("out")))
        result = CliRunner().invoke(
            cli=nrtk_perturber_cli,
            args=[
                f"--dataset_dir={tmpdir}",
                "--input_dataset_format=COCO",
                f"--output_dir={output_dir}",
                "--output_dataset_format=COCO",
                f"--config_file={config_file}",
                "-v",
            ],
        )
        assert result.exit_code == 102

    @mock.patch(f"{_CLI}.from_config_dict")
    @mock.patch(f"{_CLI}.load")
    def test_unsupported_task(
        self,
        load_patch: MagicMock,
        from_config_dict_patch: MagicMock,  # noqa: ARG002
        config_file: Path,
        tmpdir: py.path.local,
    ) -> None:
        """Check that the CLI exits with code 103 for a task with no perturber path."""
        load_patch.return_value = _fake_dataset(task=Task.IC)

        output_dir = Path(tmpdir.join(Path("out")))
        result = CliRunner().invoke(
            cli=nrtk_perturber_cli,
            args=[
                f"--dataset_dir={tmpdir}",
                "--input_dataset_format=YOLO",
                f"--output_dir={output_dir}",
                "--output_dataset_format=YOLO",
                f"--config_file={config_file}",
                "-v",
            ],
        )
        assert result.exit_code == 103

    @mock.patch(f"{_CLI}.from_config_dict")
    @mock.patch(f"{_CLI}.load")
    def test_experimental_flag(
        self,
        load_patch: MagicMock,
        from_config_dict_patch: MagicMock,  # noqa: ARG002
        config_file: Path,
        tmpdir: py.path.local,
    ) -> None:
        """Check that CLI exits with code 104 if experimental features are used without --enable-experimental."""
        load_patch.return_value = _fake_dataset(task=Task.MOT)

        output_dir = Path(tmpdir.join(Path("out")))
        result = CliRunner().invoke(
            cli=nrtk_perturber_cli,
            args=[
                f"--dataset_dir={tmpdir}",
                "--input_dataset_format=hmie",
                f"--output_dir={output_dir}",
                "--output_dataset_format=hmie",
                f"--config_file={config_file}",
                "-v",
            ],
        )
        assert result.exit_code == 104

    def test_load_failure_unknown_input_format(
        self,
        config_file: Path,
        tmpdir: py.path.local,
    ) -> None:
        """Check that CLI exits with code 105 if output_dataset_format is invalid."""
        output_dir = Path(tmpdir.join(Path("out")))

        result = CliRunner().invoke(
            cli=nrtk_perturber_cli,
            args=[
                f"--dataset_dir={tmpdir}",
                "--input_dataset_format=not_a_real_format",
                f"--output_dir={output_dir}",
                "--output_dataset_format=COCO",
                f"--config_file={config_file}",
            ],
        )
        assert result.exit_code == 105

    def test_load_failure_unknown_output_format(
        self,
        config_file: Path,
        tmpdir: py.path.local,
    ) -> None:
        """Check that CLI exits with code 106 if output_dataset_format is invalid."""
        output_dir = Path(tmpdir.join(Path("out")))

        result = CliRunner().invoke(
            cli=nrtk_perturber_cli,
            args=[
                f"--dataset_dir={tmpdir}",
                "--input_dataset_format=COCO",
                f"--output_dir={output_dir}",
                "--output_dataset_format=not_a_real_format",
                f"--config_file={config_file}",
            ],
        )
        assert result.exit_code == 106

    @mock.patch(f"{_CLI}.from_config_dict")
    @mock.patch(f"{_CLI}.load")
    def test_load_failure_incompatible_output_format(
        self,
        load_patch: MagicMock,
        from_config_dict_patch: MagicMock,  # noqa: ARG002
        config_file: Path,
        tmpdir: py.path.local,
    ) -> None:
        """Check that CLI exits with code 106 if output_dataset_format is incompatible with input_dataset_format."""
        load_patch.return_value = _fake_dataset(task=Task.OD)
        output_dir = Path(tmpdir.join(Path("out")))

        result = CliRunner().invoke(
            cli=nrtk_perturber_cli,
            args=[
                f"--dataset_dir={tmpdir}",
                "--input_dataset_format=COCO",
                f"--output_dir={output_dir}",
                "--output_dataset_format=HMIE",
                f"--config_file={config_file}",
            ],
        )
        assert result.exit_code == 106

    def test_od_regression(
        self,
        snapshot: SnapshotAssertion,
        psnr_tiff_snapshot: SnapshotAssertion,
        tmp_path: Path,
    ) -> None:
        """End-to-end OD run over a real COCO dataset, compared against snapshots."""
        snapshot = snapshot.use_extension(JSONSnapshotExtension)
        output_dir = tmp_path / "out"
        result = CliRunner().invoke(
            cli=nrtk_perturber_cli,
            args=[
                f"--dataset_dir={DATASET_FOLDER}",
                "--input_dataset_format=COCO",
                f"--output_dir={output_dir}",
                "--output_dataset_format=COCO",
                f"--config_file={NRTK_BRIGHTNESS_CONFIG}",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        img_filenames = [
            Path("0000006_02616_d_0000007.png"),
            Path("0000006_03636_d_0000009.png"),
            Path("0000006_00159_d_0000001.png"),
            Path("0000006_01659_d_0000004.png"),
            Path("0000161_01584_d_0000158.png"),
            Path("0000006_01111_d_0000003.png"),
            Path("0000006_04050_d_0000010.png"),
            Path("0000006_04309_d_0000011.png"),
            Path("0000006_01275_d_0000004.png"),
            Path("0000006_00611_d_0000002.png"),
            Path("0000006_02138_d_0000006.png"),
        ]

        dataset_dir = output_dir / "_factor-1"

        # Check annotations
        annotations_path = dataset_dir / "annotations" / "instances.json"
        assert annotations_path.exists()
        with open(annotations_path, encoding="utf-8") as annotations:
            snapshot.assert_match(json.load(annotations))

        # Check all images written out
        for img_filename in img_filenames:
            img_path = dataset_dir / img_filename
            assert img_path.exists()
            with Image.open(img_path) as img:
                psnr_tiff_snapshot.assert_match(np.asarray(img))

    def test_od_combine_ouput_regression(
        self,
        snapshot: SnapshotAssertion,
        psnr_tiff_snapshot: SnapshotAssertion,
        tmp_path: Path,
    ) -> None:
        """End-to-end OD run over a real COCO dataset, compared against snapshots."""
        snapshot = snapshot.use_extension(JSONSnapshotExtension)
        output_dir = tmp_path / "out"
        result = CliRunner().invoke(
            cli=nrtk_perturber_cli,
            args=[
                f"--dataset_dir={DATASET_FOLDER}",
                "--input_dataset_format=COCO",
                f"--output_dir={output_dir}",
                "--output_dataset_format=COCO",
                f"--config_file={NRTK_BRIGHTNESS_CONFIG}",
                "--combine_output",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        img_filenames = [
            Path("0.png"),
            Path("1.png"),
            Path("2.png"),
            Path("3.png"),
            Path("4.png"),
            Path("5.png"),
            Path("6.png"),
            Path("7.png"),
            Path("8.png"),
            Path("9.png"),
            Path("10.png"),
        ]

        # Check annotations
        annotations_path = output_dir / "annotations" / "instances.json"
        assert annotations_path.exists()
        with open(annotations_path, encoding="utf-8") as annotations:
            snapshot.assert_match(json.load(annotations))

        # Check all images written out
        images_dir = output_dir / "images"
        for img_filename in img_filenames:
            img_path = images_dir / img_filename
            assert img_path.exists()
            with Image.open(img_path) as img:
                psnr_tiff_snapshot.assert_match(np.asarray(img))

    def test_mot_regression(
        self,
        snapshot: SnapshotAssertion,
        psnr_mp4_snapshot: SnapshotAssertion,
        tmp_path: Path,
    ) -> None:
        """End-to-end MOT run over a real (annotation-free) video, compared against snapshots."""
        snapshot = snapshot.use_extension(JSONSnapshotExtension)
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        shutil.copy(src=INPUT_DRONE_VIDEO_FILE_PATH, dst=input_dir / "drone_clip.mp4")
        output_dir = tmp_path / "out"

        result = CliRunner().invoke(
            cli=nrtk_perturber_cli,
            args=[
                f"--dataset_dir={input_dir}",
                "--input_dataset_format=flat_mp4",
                f"--output_dir={output_dir}",
                "--output_dataset_format=hmie",
                f"--config_file={NRTK_BRIGHTNESS_CONFIG}",
                "--enable_experimental",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        dataset_dir = output_dir / "_factor-1"

        snippet_dir = dataset_dir / "out_000000_000000" / "out_000000_000001"

        # Check annotations
        annotations_path = snippet_dir / "scale" / "CDAO_OUT_out_000000_000001.mp4_rt.json"
        assert annotations_path.exists()
        annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
        # source_path changes each execution. Check that exists and then delete before snapshot
        assert "source_path" in annotations
        del annotations["source_path"]
        snapshot.assert_match(annotations)

        # Check the perturbed video (first frames only)
        video_path = snippet_dir / "seq_mp4" / "out_000000_000001.mp4"
        assert video_path.exists()
        psnr_mp4_snapshot.assert_match(itertools.islice(read_video(str(video_path)), 5))  # noqa: FKA100 - islice is a C function

    def test_mot_combine_output_regression(
        self,
        snapshot: SnapshotAssertion,
        psnr_mp4_snapshot: SnapshotAssertion,
        tmp_path: Path,
    ) -> None:
        """End-to-end MOT run over a real (annotation-free) video, compared against snapshots."""
        snapshot = snapshot.use_extension(JSONSnapshotExtension)
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        shutil.copy(src=INPUT_DRONE_VIDEO_FILE_PATH, dst=input_dir / "drone_clip.mp4")
        output_dir = tmp_path / "out"

        result = CliRunner().invoke(
            cli=nrtk_perturber_cli,
            args=[
                f"--dataset_dir={input_dir}",
                "--input_dataset_format=flat_mp4",
                f"--output_dir={output_dir}",
                "--output_dataset_format=hmie",
                f"--config_file={NRTK_BRIGHTNESS_CONFIG}",
                "--combine_output",
                "--enable_experimental",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        snippet_dir = output_dir / "out_000000_000000" / "out_000000_000001"

        # Check annotations
        annotations_path = snippet_dir / "scale" / "CDAO_OUT_out_000000_000001.mp4_rt.json"
        assert annotations_path.exists()
        annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
        # source_path changes each execution. Check that exists and then delete before snapshot
        assert "source_path" in annotations
        del annotations["source_path"]
        snapshot.assert_match(annotations)

        # Check the perturbed video (first frames only)
        video_path = snippet_dir / "seq_mp4" / "out_000000_000001.mp4"
        assert video_path.exists()
        psnr_mp4_snapshot.assert_match(itertools.islice(read_video(str(video_path)), 5))  # noqa: FKA100 - islice is a C function

    def test_load_metadata_preserves_id_types(self) -> None:
        """Load JSON metadata while keeping integer and string IDs distinct."""
        metadata = _load_metadata(dataset_dir=str(DATUM_METADATA.parent))

        assert metadata == {
            1: {"img_gsd": 0.25},
            "1": {"img_gsd": 0.50},
            "frame-a": {"img_gsd": 0.75},
        }
