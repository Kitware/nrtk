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


class PyBSMAndOpenCVImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__(
            "pyBSM and OpenCV must be installed. Please install via `nrtk[pybsm,graphics]` or `nrtk[pybsm,headless]`.",
        )


class PyBSMImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__("pyBSM must be installed. Please install via `nrtk[pybsm]`.")


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


class AlbumentationsImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__("albumentations must be installed. Please install via `nrtk[albumentations]`.")
