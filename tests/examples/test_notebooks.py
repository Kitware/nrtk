import sys

import pytest

from .test_notebook_utils import list_error_messages, pyright_analyze


@pytest.mark.notebooks
# require_marker here is an execution gate (needs pyright; only run under -m notebooks),
# not the dependency-canary use seen in the *_optional_deps.py files.
@pytest.mark.usefixtures("require_marker")
@pytest.mark.parametrize(
    ("filepath", "expected_num_errors"),
    [
        ("docs/examples/albumentations_perturber.ipynb", 0),
        ("docs/examples/generative_perturbers.ipynb", 0),
        ("docs/examples/end_to_end_overview.ipynb", 0),
        ("docs/examples/misb_st1909_burn_in_perturber.ipynb", 0),
        ("docs/examples/optical_perturbers.ipynb", 0),
        ("docs/examples/photometric_perturbers.ipynb", 0),
        ("docs/examples/pybsm_default_config.ipynb", 0),
        ("docs/examples/maite/affine_transformations.ipynb", 0),
        ("docs/examples/maite/extreme_illumination.ipynb", 0),
        ("docs/examples/maite/visual_focus.ipynb", 0),
        ("docs/examples/maite/fog_haze.ipynb", 0),
        ("docs/examples/maite/motion_jitter.ipynb", 0),
        ("docs/examples/maite/lens_flare.ipynb", 0),
        ("docs/examples/maite/radial_distortion.ipynb", 0),
        ("docs/examples/maite/sensor_resolution_and_noise.ipynb", 0),
        ("docs/examples/maite/atmospheric_turbulence.ipynb", 0),
        ("docs/examples/maite/water_droplets.ipynb", 0),
        # Issue with trackeval and numpy version requirements produce a type error on
        # 3.12 and 3.13
        pytest.param(
            "docs/examples/maite/framewise_video.ipynb",
            1,
            marks=pytest.mark.skipif(
                sys.version_info >= (3, 14),
                reason="framewise_video notebook dependencies are not yet supported on Python 3.14",
            ),
        ),
        # https://gitlab.jatic.net/jatic/kitware/nrtk/-/issues/698
        # ("docs/examples/xaitk_saliency_workflow/image_classification_perturbation_saliency.ipynb", 0),
        # ("docs/examples/xaitk_saliency_workflow/object_detection_perturbation_saliency.ipynb", 0),
    ],
)
def test_pyright_nb(filepath: str, expected_num_errors: int) -> None:
    results = pyright_analyze(notebook_path_str=filepath)  # type: ignore
    assert results["summary"]["errorCount"] <= expected_num_errors, list_error_messages(results)  # type: ignore
