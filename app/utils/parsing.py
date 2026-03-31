from __future__ import annotations

import re
from pathlib import Path


def safe_prefix(prefix: str | None, source_path: str, tool: str = "") -> str:
    """Sanitize a file prefix for tool output paths.

    If *prefix* is ``None``, a default is derived from the stem of
    *source_path* with an optional *.tool* suffix.
    """
    suffix = f".{tool}" if tool else ""
    raw = prefix or f"{Path(source_path).stem}{suffix}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw)


def maybe_float(value: str | None) -> float | None:
    """Parse a nullable string to float, treating blank/NA as ``None``."""
    if value is None:
        return None
    text = value.strip()
    if not text or text.upper() == "NA":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def maybe_int(value: str | None) -> int | None:
    """Parse a nullable string to int, treating blank/NA as ``None``."""
    if value is None:
        return None
    text = value.strip()
    if not text or text.upper() == "NA":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None
