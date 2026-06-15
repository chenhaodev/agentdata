"""Normalize any source row into canonical DataItems."""

from .detect import detect_format
from .normalize import normalize_row, normalize_rows

__all__ = ["detect_format", "normalize_row", "normalize_rows"]
