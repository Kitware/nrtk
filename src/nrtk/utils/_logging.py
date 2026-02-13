__all__ = ["setup_logging"]

import logging
import time

try:
    from pythonjsonlogger.json import JsonFormatter
except ImportError:
    JsonFormatter = None


def setup_logging(*, name: str, loglevel: int = logging.WARN) -> logging.Logger:
    """Utility function to configure json logging."""
    logging.Formatter.converter = time.gmtime
    logger = logging.getLogger(name)

    logger.setLevel(loglevel)
    if JsonFormatter is not None:
        log_handler = logging.StreamHandler()
        formatter = JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
        log_handler.setFormatter(formatter)
        logger.addHandler(log_handler)

    return logger
