class AlbumentationsImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__("albumentations must be installed. Please install via `nrtk[albumentations]`.")


class DiffusionImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__("torch, diffusers, and PIL must be installed. Please install via `nrtk[diffusion]`.")


class FastApiImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__("fastapi must be installed. Please install via `nrtk[maite]`.")


class KWCocoImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__("kwcoco must be installed. Please install via `nrtk[tools]`.")


class MaiteImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__("maite must be installed. Please install via `nrtk[maite]`.")


class OpenCVImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__("OpenCV must be installed. Please install via `nrtk[graphics]` or `nrtk[headless]`.")


class PillowImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__("PIL must be installed. Please install via `nrtk[Pillow]`.")


class PyBSMImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__("pyBSM must be installed. Please install via `nrtk[pybsm]`.")


class ScipyImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__("scipy must be installed. Please install via `nrtk[waterdroplet]`.")


class TorchImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__("torchvision must be installed. Please install via `nrtk[notebook-testing]`.")


class PydanticImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__("pydantic must be installed. Please install via `nrtk[maite]`.")


class PydanticSettingsImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__("pydantic_settings must be installed. Please install via `nrtk[maite]`.")


class ScikitImageImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__("scikit-image must be installed. Please install via `nrtk[scikit-image]`.")


class WaterDropletImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__("""Scipy, Shapely and GeoPandas must be installed. Please install via nrtk[waterdroplet]`.""")


class NotebookTestingImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__("jupytext must be installed. Please install via `nrtk[notebook-testing]`.")


class NRTKXAITKHelperImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__(
            """
            Helper functions for the NRTK-XAITK workflow must be installed.
            Please install via `nrtk[maite,Pillow,notebook-testing]`.
            """,
        )
