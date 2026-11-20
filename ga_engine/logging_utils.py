"""Logging setup that follows library best practice.

``ga_engine`` never configures logging on its own. All modules log to children
of the ``"ga_engine"`` logger, which carries a :class:`logging.NullHandler`.
Applications choose where records go; :func:`enable_console_logging` is a
convenience (also behind the ``verbose=True`` engine flag) that is idempotent,
so repeated calls can never produce duplicated output lines.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional, TextIO

PACKAGE_LOGGER_NAME = "ga_engine"

_CONSOLE_MARK = "_ga_engine_console_handler"
_LOCK = threading.Lock()

logging.getLogger(PACKAGE_LOGGER_NAME).addHandler(logging.NullHandler())


def enable_console_logging(
    level: int = logging.INFO, stream: Optional[TextIO] = None
) -> logging.Handler:
    """Attach (at most) one console handler to the ``ga_engine`` logger.

    Calling this repeatedly reuses the existing handler and only updates the
    level, so it is safe to call from many engine instances and threads.

    Args:
        level: Logging level for the package logger and the handler.
        stream: Destination stream; defaults to ``sys.stderr``. Ignored when a
            handler is already installed.

    Returns:
        The installed console handler.
    """
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    with _LOCK:
        handler = next(
            (h for h in package_logger.handlers if getattr(h, _CONSOLE_MARK, False)),
            None,
        )
        if handler is None:
            handler = logging.StreamHandler(stream)
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            setattr(handler, _CONSOLE_MARK, True)
            package_logger.addHandler(handler)
        handler.setLevel(level)
        package_logger.setLevel(level)
    return handler
