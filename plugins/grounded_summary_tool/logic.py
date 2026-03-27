from __future__ import annotations

from app.models import AnalysisFacts, RecommendationItem, ReferenceItem, VariantAnnotation
from app.services.annotation import build_draft_answer


def execute(payload: dict[str, object]) -> dict[str, object]:
    facts = AnalysisFacts(**payload["facts"])
    annotations = [VariantAnnotation(**item) for item in payload.get("annotations", [])]
    references = [ReferenceItem(**item) for item in payload.get("references", [])]
    recommendations = [RecommendationItem(**item) for item in payload.get("recommendations", [])]

    draft_answer = build_draft_answer(
        facts,
        annotations,
        [item.id for item in references],
        [item.id for item in recommendations],
    )
    return {"draft_answer": draft_answer}
