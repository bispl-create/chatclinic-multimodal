"""Dementia prediction plugin (remote inference).

Calls the Dementia-R1 API (fine-tuned Qwen2) over HTTP. The API server is
expected to run on a GPU machine and expose a ``POST /predict`` endpoint that
accepts ``{"clinical_notes": [str, ...]}`` and returns ``predictions``,
``summary``, and ``elapsed_sec``.

The target URL is configurable via the ``DEMENTIA_API_URL`` environment
variable (default ``http://127.0.0.1:8020``). Use an SSH tunnel when the GPU
server is remote.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

DEMENTIA_API_URL = os.environ.get("DEMENTIA_API_URL", "http://127.0.0.1:8020")


def _call_predict(clinical_notes: list[str]) -> dict:
    url = f"{DEMENTIA_API_URL.rstrip('/')}/predict"
    body = json.dumps({"clinical_notes": clinical_notes}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Dementia API at {url}. "
            f"Is the inference server running (and the SSH tunnel open, if remote)? "
            f"Error: {exc}"
        ) from exc


def execute(payload: dict[str, object]) -> dict[str, object]:
    text_path = str(payload.get("text_path") or "").strip()
    if not text_path:
        raise ValueError("`text_path` is required. Upload a clinical note first.")
    path = Path(text_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Clinical note not found: {path}")

    clinical_text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not clinical_text:
        raise ValueError("The uploaded clinical note is empty.")

    api_result = _call_predict([clinical_text])
    predictions = api_result.get("predictions", [])

    return {
        "dementia_prediction": {
            "summary": api_result.get("summary", "Prediction completed."),
            "predictions": predictions,
            "num_patients": len(predictions),
            "inference_time_sec": api_result.get("elapsed_sec"),
            "model": "Dementia-R1 (Qwen2 fine-tuned)",
            "inference_server": DEMENTIA_API_URL,
        },
    }
