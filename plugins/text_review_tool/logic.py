from __future__ import annotations

import mimetypes
from pathlib import Path

from app.models import TextSourceResponse
from app.services.tool_runner import discover_tools


def analyze_text_source(text_path: str, file_name: str | None = None) -> TextSourceResponse:
    path = Path(text_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Text source not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    preview_lines = [line.strip() for line in lines if line.strip()][:12]
    media_type = mimetypes.guess_type(path.name)[0] or "text/plain"
    warnings: list[str] = []
    if not text.strip():
        warnings.append("The uploaded text source is empty.")
    if len(text) > 100_000:
        warnings.append("Only the leading portion of this text may be shown in preview cards.")
    draft_answer = (
        f"Text review is ready for `{file_name or path.name}`.\n\n"
        f"- Characters: {len(text)}\n"
        f"- Words: {len(text.split())}\n"
        f"- Lines: {len(lines)}\n\n"
        "The Studio card now shows a preview of the uploaded note. Ask follow-up questions or use `$studio ...` for grounded explanation of the current text source."
    )
    return TextSourceResponse(
        analysis_id="",
        source_text_path=str(path),
        file_name=file_name or path.name,
        media_type=media_type,
        char_count=len(text),
        word_count=len(text.split()),
        line_count=len(lines),
        preview_lines=preview_lines,
        warnings=warnings,
        draft_answer=draft_answer,
        used_tools=["text_review_tool"],
        tool_registry=discover_tools(),
    )


def execute(payload: dict[str, object]) -> dict[str, object]:
    text_path = str(payload.get("text_path") or "").strip()
    if not text_path:
        raise ValueError("`text_path` is required.")
    file_name = str(payload.get("file_name") or Path(text_path).name).strip()
    analysis = analyze_text_source(text_path, file_name=file_name)
    return {"analysis": analysis.model_dump()}
