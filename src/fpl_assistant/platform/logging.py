from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=level.upper(),
        format="%(levelname)s | %(asctime)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("fpl_assistant")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

