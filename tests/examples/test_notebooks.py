import pytest

from nrtk.utils._exceptions import NotebookTestingImportError
from nrtk.utils._import_guard import is_available

from .test_notebook_utils import list_error_messages, pyright_analyze

deps = ["jupytext"]
is_usable = all(is_available(dep) for dep in deps)


@pytest.mark.skipif(not is_usable, reason=str(NotebookTestingImportError()))
class TestPyrightNotebook:
    @pytest.mark.filterwarnings("ignore:Jupyter is migrating its paths")
    @pytest.mark.parametrize(
        ("filepath", "expected_num_errors"),
        [
            # ("docs/examples/maite/gradio/nrtk-gradio.ipynb", 0), Temporarily removed until sensor/scenario are removed
            ("docs/examples/maite/augmentations.ipynb", 0),
            # Suppressing type checking errors related to xaitk-* imports
            ("docs/examples/maite/jatic-perturbations-saliency.ipynb", 3),
            ("docs/examples/maite/nrtk_brightness_perturber_demo.ipynb", 0),
            ("docs/examples/maite/nrtk_haze_perturber_demo.ipynb", 0),
            ("docs/examples/maite/nrtk_focus_perturber_demo.ipynb", 0),
            ("docs/examples/maite/nrtk_sensor_transformation_demo.ipynb", 0),
            ("docs/examples/maite/nrtk_lens_flare_demo.ipynb", 0),
            ("docs/examples/maite/nrtk_affine_perturbers_demo.ipynb", 0),
            ("docs/examples/maite/nrtk_jitter_perturber_demo.ipynb", 0),
            ("docs/examples/maite/nrtk_turbulence_perturber_demo.ipynb", 0),
            ("docs/examples/maite/nrtk_water_droplet_perturber_demo.ipynb", 0),
            ("docs/examples/maite/nrtk_radial_distortion_perturber_demo.ipynb", 0),
            ("docs/examples/pybsm/pybsm_test.ipynb", 0),
            ("docs/examples/coco_scorer.ipynb", 0),
            ("docs/examples/nrtk_tutorial.ipynb", 0),
            ("docs/examples/otf_visualization.ipynb", 0),
            ("docs/examples/perturbers.ipynb", 0),
            ("docs/examples/simple_generic_generator.ipynb", 0),
            ("docs/examples/simple_pybsm_generator.ipynb", 0),
            ("docs/examples/albumentations_perturber.ipynb", 0),
        ],
    )
    def test_pyright_nb(self, filepath: str, expected_num_errors: int) -> None:
        results = pyright_analyze(filepath)  # type: ignore
        assert results["summary"]["errorCount"] <= expected_num_errors, list_error_messages(results)  # type: ignore
