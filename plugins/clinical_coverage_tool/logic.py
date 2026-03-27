from __future__ import annotations


def _has_meaningful_text(value: str | None) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text and text not in {".", "n/a", "NA"})


def _detail(label: str, count: int, total: int) -> dict[str, object]:
    percent = round((count / total) * 100) if total else 0
    return {"label": label, "count": count, "detail": f"{count}/{total} annotated ({percent}%)"}


def execute(payload: dict[str, object]) -> dict[str, object]:
    annotations = list(payload.get("annotations", []))
    total = len(annotations)
    summary = [
        _detail(
            "ClinVar coverage",
            sum(
                1
                for item in annotations
                if _has_meaningful_text(item.get("clinical_significance")) or _has_meaningful_text(item.get("clinvar_conditions"))
            ),
            total,
        ),
        _detail("gnomAD coverage", sum(1 for item in annotations if _has_meaningful_text(item.get("gnomad_af"))), total),
        _detail("Gene mapping", sum(1 for item in annotations if _has_meaningful_text(item.get("gene"))), total),
        _detail(
            "HGVS coverage",
            sum(1 for item in annotations if _has_meaningful_text(item.get("hgvsc")) or _has_meaningful_text(item.get("hgvsp"))),
            total,
        ),
        _detail("Protein change", sum(1 for item in annotations if _has_meaningful_text(item.get("hgvsp"))), total),
    ]
    return {"clinical_coverage_summary": summary}
