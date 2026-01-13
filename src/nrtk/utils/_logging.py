import logging
import time

from nrtk.utils._exceptions import MaiteImportError
from nrtk.utils._import_guard import import_guard

json_logging_available: bool = import_guard(
    module_name="pythonjsonlogger",
    exception=MaiteImportError,
    submodules=["json"],
    objects=["JSONFormatter"],
)
from pythonjsonlogger.json import JsonFormatter  # noqa: E402


def setup_logging(*, name: str, loglevel: int = logging.WARN) -> logging.Logger:
    """Utility function to configure json logging."""
    logging.Formatter.converter = time.gmtime
    logger = logging.getLogger(name)

    logger.setLevel(loglevel)
    if json_logging_available:
        log_handler = logging.StreamHandler()
        formatter = JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
        log_handler.setFormatter(formatter)
        logger.addHandler(log_handler)

    return logger
