from __future__ import annotations

import logging

import structlog

from quantagent.config import settings


def configure_logging() -> None:
    """Wire structlog so every log line carries bound context (e.g. `trace_id`)
    and renders as JSON in non-local environments. Call once at process startup;
    safe to call more than once.
    """
    logging.basicConfig(format="%(message)s", level=settings.log_level)
    renderer = (
        structlog.dev.ConsoleRenderer()
        if settings.env == "local"
        else structlog.processors.JSONRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
