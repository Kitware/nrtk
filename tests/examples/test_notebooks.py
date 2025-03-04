import pytest

from nrtk.utils._exceptions import MaiteImportError

maite_available = True
try:
    from maite.testing.pyright import list_error_messages, pyright_analyze
except ImportError:  # pragma: no cover
    maite_available = False


@pytest.mark.skipif(not maite_available, reason=str(MaiteImportError()))
class TestPyrightNotebook:
    @pytest.mark.filterwarnings("ignore:Jupyter is migrating its paths")
    @pytest.mark.parametrize(
        ("filepath", "expected_num_errors"),
        [
            ("docs/examples/maite/gradio/nrtk-gradio.ipynb", 0),
            ("docs/examples/maite/augmentations.ipynb", 0),
            ("docs/examples/maite/jatic-perturbations-saliency.ipynb", 0),
            ("docs/examples/maite/nrtk_brightness_perturber_demo.ipynb", 0),
            ("docs/examples/maite/nrtk_focus_perturber_demo.ipynb", 0),
            ("docs/examples/maite/nrtk_sensor_transformation_demo.ipynb", 0),
            ("docs/examples/maite/nrtk_translation_perturber_demo.ipynb", 0),
            ("docs/examples/maite/nrtk_turbulence_perturber_demo.ipynb", 0),
            ("docs/examples/pybsm/pybsm_test.ipynb", 0),
            ("docs/examples/coco_scorer.ipynb", 0),
            ("docs/examples/nrtk_tutorial.ipynb", 0),
            ("docs/examples/otf_visualization.ipynb", 0),
            ("docs/examples/perturbers.ipynb", 0),
            ("docs/examples/simple_generic_generator.ipynb", 0),
            ("docs/examples/simple_pybsm_generator.ipynb", 0),
        ],
    )
    def test_pyright_nb(self, filepath: str, expected_num_errors: int) -> None:
        results = pyright_analyze(filepath)[0]  # type: ignore
        assert results["summary"]["errorCount"] <= expected_num_errors, list_error_messages(results)  # type: ignore
