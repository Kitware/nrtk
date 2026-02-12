import pytest

from .test_notebook_utils import list_error_messages, pyright_analyze


@pytest.mark.notebooks
@pytest.mark.parametrize(
    ("filepath", "expected_num_errors"),
    [
        ("docs/examples/albumentations_perturber.ipynb", 0),
        ("docs/examples/generative_perturbers.ipynb", 0),
        ("docs/examples/nrtk_tutorial.ipynb", 0),
        ("docs/examples/optical_perturbers.ipynb", 0),
        ("docs/examples/photometric_perturbers.ipynb", 0),
        ("docs/examples/pybsm_default_config.ipynb", 0),
        ("docs/examples/maite/jatic-perturbations-saliency.ipynb", 0),
        ("docs/examples/maite/nrtk_affine_perturbers_demo.ipynb", 0),
        ("docs/examples/maite/nrtk_brightness_perturber_demo.ipynb", 0),
        ("docs/examples/maite/nrtk_focus_perturber_demo.ipynb", 0),
        ("docs/examples/maite/nrtk_haze_perturber_demo.ipynb", 0),
        ("docs/examples/maite/nrtk_jitter_perturber_demo.ipynb", 0),
        ("docs/examples/maite/nrtk_lens_flare_demo.ipynb", 0),
        ("docs/examples/maite/nrtk_radial_distortion_perturber_demo.ipynb", 0),
        ("docs/examples/maite/nrtk_sensor_transformation_demo.ipynb", 0),
        ("docs/examples/maite/nrtk_turbulence_perturber_demo.ipynb", 0),
        ("docs/examples/maite/nrtk_water_droplet_perturber_demo.ipynb", 0),
        # https://gitlab.jatic.net/jatic/kitware/nrtk/-/issues/698
        # ("docs/examples/nrtk_xaitk_workflow/image_classification_perturbation_saliency.ipynb", 0),
        # ("docs/examples/nrtk_xaitk_workflow/object_detection_perturbation_saliency.ipynb", 0),
    ],
)
def test_pyright_nb(filepath: str, expected_num_errors: int) -> None:
    results = pyright_analyze(notebook_path_str=filepath)  # type: ignore
    assert results["summary"]["errorCount"] <= expected_num_errors, list_error_messages(results)  # type: ignore
