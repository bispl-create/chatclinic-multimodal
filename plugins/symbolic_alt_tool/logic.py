from __future__ import annotations


def execute(payload: dict[str, object]) -> dict[str, object]:
    annotations = list(payload.get("annotations", []))
    symbolic = [
        item
        for item in annotations
        if any(str(alt).startswith("<") and str(alt).endswith(">") for alt in item.get("alts", []))
    ]
    summary = {
        "count": len(symbolic),
        "examples": [
            {
                "locus": f"{item.get('contig')}:{item.get('pos_1based')}",
                "gene": item.get("gene") or "",
                "alts": item.get("alts", []),
                "consequence": item.get("consequence") or "",
                "genotype": item.get("genotype") or "",
            }
            for item in symbolic[:5]
        ],
    }
    return {"symbolic_alt_summary": summary}
