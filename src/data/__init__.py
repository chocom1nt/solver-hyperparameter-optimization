"""Data loading and metadata extraction."""

from .metadata_extractor import extract_metadata
from .task_loader import load_tasks

__all__ = ["extract_metadata", "load_tasks"]

