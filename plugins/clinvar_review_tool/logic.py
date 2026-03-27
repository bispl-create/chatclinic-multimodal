from __future__ import annotations

from collections import Counter


def _label(text: str, fallback: str) -> str:
    value = (text or "").strip()
    return value if value and value != "." else fallback


def execute(payload: dict[str, object]) -> dict[str, object]:
    annotations = list(payload.get("annotations", []))
    counts = Counter(_label(item.get("clinical_significance", ""), "Unreviewed") for item in annotations)
    summary = [{"label": label, "count": count} for label, count in counts.most_common()]
    return {"clinvar_summary": summary}
