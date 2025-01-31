class OpenCVImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__("OpenCV must be installed. Please install via `nrtk[graphics]` or `nrtk[headless]`.")


class PillowImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__("PIL must be installed. Please install via `nrtk[Pillow]`.")


class ScikitImageImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__("scikit-image must be installed. Please install via `nrtk[scikit-image]`.")


class PyBSMImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__("pyBSM must be installed. Please install via `nrtk[pybsm]`.")


class KWCocoImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__("kwcoco must be installed. Please install via `nrtk[tools]`.")


class PyBSMAndOpenCVImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__(
            "pyBSM and OpenCV must be installed. Please install via `nrtk[pybsm,graphics]` or `nrtk[pybsm,headless]`.",
        )


class MaiteImportError(ImportError):
    def __init__(self) -> None:
        # Call the base class constructor with the parameters it needs
        super().__init__("maite must be installed. Please install via `nrtk[maite]`.")
