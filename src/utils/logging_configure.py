"""Logging configuration utilities for the experiment framework.

This module provides centralized logging setup that writes logs to both
console and a rotating log file.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def setup_logging(
    log_dir: str | Path | None = None,
    log_file: str | None = None,
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True,
) -> None:
    """Configure logging for the entire application.

    Args:
        log_dir: Directory for log files. Defaults to 'logs' in current dir.
        log_file: Base name for log file. If None, auto-generates name.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        console: Whether to log to console.
        file: Whether to log to file.
    """
    # Create log directory if needed
    if log_dir is None:
        log_dir = Path("logs")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate log filename if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"experiment_{timestamp}.log"
    log_path = log_dir / log_file

    # Get root logger and clear existing handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    # Formatter for all handlers
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler with UTF-8 encoding
    if file:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Log initial message
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized. Log file: %s", log_path)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    This is a convenience function that ensures the logger uses
    the naming convention for the project (src.* modules).

    Args:
        name: Logger name (e.g., 'scip_solver' or 'experiment.runner').

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


class LoggingContext:
    """Context manager for temporary logging configuration changes."""

    def __init__(self, level: int = logging.DEBUG):
        self.level = level
        self.original_level: int | None = None

    def __enter__(self) -> "LoggingContext":
        root_logger = logging.getLogger()
        self.original_level = root_logger.level
        root_logger.setLevel(self.level)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.original_level is not None:
            logging.getLogger().setLevel(self.original_level)
