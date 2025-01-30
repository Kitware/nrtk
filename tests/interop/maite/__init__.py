from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATASET_FOLDER = DATA_DIR / "VisDrone2019-DET-test-dev-TINY"
LABEL_FILE = DATASET_FOLDER / "annotations.json"
NRTK_PYBSM_CONFIG = DATA_DIR / "nrtk_pybsm_config.json"
NRTK_BLUR_CONFIG = DATA_DIR / "nrtk_blur_config.json"
BAD_NRTK_CONFIG = DATA_DIR / "nrtk_bad_config.json"
EMPTY_NRTK_CONFIG = DATA_DIR / "nrtk_empty_config.json"
