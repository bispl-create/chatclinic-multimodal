from __future__ import annotations

from collections import Counter


def _label(text: str, fallback: str) -> str:
    value = (text or "").strip()
    return value if value and value != "." else fallback


def execute(payload: dict[str, object]) -> dict[str, object]:
    annotations = list(payload.get("annotations", []))
    limit = int(payload.get("limit", 10))
    counts = Counter(_label(item.get("consequence", ""), "Unclassified") for item in annotations)
    summary = [{"label": label, "count": count} for label, count in counts.most_common(limit)]
    return {"consequence_summary": summary}
