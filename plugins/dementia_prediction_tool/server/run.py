#!/usr/bin/env python3
"""
Dementia Prediction Tool for ChatClinic
========================================
Predicts dementia likelihood from longitudinal clinical notes using
the fine-tuned Qwen2 Dementia-R1 model via vLLM.

Input:  clinical notes (plain text / markdown) via ChatClinic payload
Output: binary prediction (0/1), probability score, reasoning
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = os.environ.get(
    "DEMENTIA_MODEL_PATH",
    str(Path(__file__).resolve().parents[2].parent / "Asan" / "Dementia-R1" / "checkpoint-4600"),
)
MAX_TOKENS = int(os.environ.get("DEMENTIA_MAX_TOKENS", "1024"))
MAX_MODEL_LEN = int(os.environ.get("DEMENTIA_MAX_MODEL_LEN", "4096"))
TENSOR_PARALLEL = int(os.environ.get("DEMENTIA_TP_SIZE", "1"))
GPU_UTIL = float(os.environ.get("DEMENTIA_GPU_UTIL", "0.90"))
GPU_DEVICE = os.environ.get("DEMENTIA_GPU_DEVICE", "0")

# Pin to specific GPU before any CUDA initialization
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_DEVICE

# ---------------------------------------------------------------------------
# Regex patterns (from the evaluation code)
# ---------------------------------------------------------------------------
BOX01_RE = re.compile(r"\\boxed\{([01])\}")
ANSWER_BLOCK_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.DOTALL | re.IGNORECASE)

# ---------------------------------------------------------------------------
# System prompt for inference
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = ""


def extract_answer_block(text: str) -> str:
    m = ANSWER_BLOCK_RE.search(text)
    return m.group(1) if m else text


def extract_label(text: str) -> Optional[str]:
    scoped = extract_answer_block(text)
    m = BOX01_RE.search(scoped)
    if m:
        return m.group(1)
    m = BOX01_RE.search(text)
    if m:
        return m.group(1)
    tokens = re.findall(r"\b([01])\b", scoped)
    if not tokens:
        tokens = re.findall(r"\b([01])\b", text)
    return tokens[-1] if tokens else None


def get_prob_score(request_output, tokenizer) -> float:
    outputs = request_output.outputs[0]
    token_ids = outputs.token_ids
    logprobs_list = outputs.logprobs

    generated_text = outputs.text
    pred_label = extract_label(generated_text)
    if not pred_label:
        return 0.5

    target_digit = pred_label
    target_idx = -1
    for i in range(len(token_ids) - 1, -1, -1):
        token_str = tokenizer.decode([token_ids[i]]).strip()
        if target_digit in token_str:
            target_idx = i
            break

    if target_idx == -1:
        return 0.5

    step_logprobs = logprobs_list[target_idx]

    prob_0 = 0.0
    prob_1 = 0.0
    for tid, lp_obj in step_logprobs.items():
        lp_val = lp_obj.logprob if hasattr(lp_obj, "logprob") else lp_obj
        decoded_t = tokenizer.decode([tid]).strip()
        if "0" in decoded_t and len(decoded_t) <= 2:
            prob_0 += math.exp(lp_val)
        if "1" in decoded_t and len(decoded_t) <= 2:
            prob_1 += math.exp(lp_val)

    if prob_0 == 0 and prob_1 == 0:
        return 0.5
    if prob_0 == 0:
        prob_0 = 1e-10
    if prob_1 == 0:
        prob_1 = 1e-10

    return prob_1 / (prob_0 + prob_1)


def build_prompt(clinical_text: str, tokenizer) -> str:
    """Build a chat-formatted prompt from clinical note text."""
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": clinical_text})

    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return (SYSTEM_PROMPT + "\n" if SYSTEM_PROMPT else "") + clinical_text


def _find_uploaded_file(file_name: str) -> Optional[str]:
    """Search ChatClinic runtime_uploads for the original uploaded file."""
    uploads_dir = Path(__file__).resolve().parents[2] / "runtime_uploads"
    if not uploads_dir.exists():
        return None
    # runtime_uploads/<uuid_hex>/<safe_filename>
    for candidate in sorted(uploads_dir.glob(f"*/{file_name}"), key=lambda p: p.stat().st_mtime, reverse=True):
        if candidate.is_file():
            return str(candidate)
    # Also try partial match (safe_filename may differ slightly)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", file_name).strip("._")
    if safe_name != file_name:
        for candidate in sorted(uploads_dir.glob(f"*/{safe_name}"), key=lambda p: p.stat().st_mtime, reverse=True):
            if candidate.is_file():
                return str(candidate)
    return None


def extract_clinical_text(payload: dict) -> list[str]:
    """
    Extract clinical note text(s) from the ChatClinic payload.

    Supports (in priority order):
      1. Find original uploaded file in runtime_uploads by file_name
      2. payload["files"] with base64-encoded text/markdown files
      3. payload["clinical_notes"] as a list of strings
      4. payload["text"] as a single string
      5. ChatClinic analysis_artifacts preview as fallback
    """
    texts: list[str] = []

    # Source 1: Find original file from runtime_uploads using file_name
    for source in payload.get("analysis_sources") or []:
        fname = source.get("file_name")
        if fname:
            found = _find_uploaded_file(fname)
            if found:
                try:
                    content = Path(found).read_text(encoding="utf-8")
                    if content.strip():
                        texts.append(content.strip())
                except Exception:
                    pass
    # Also check single source
    if not texts:
        source = payload.get("analysis_source") or {}
        fname = source.get("file_name")
        if fname:
            found = _find_uploaded_file(fname)
            if found:
                try:
                    content = Path(found).read_text(encoding="utf-8")
                    if content.strip():
                        texts.append(content.strip())
                except Exception:
                    pass

    if texts:
        return texts

    # Source 2: files with base64-encoded content
    for item in payload.get("files") or []:
        raw_b64 = item.get("raw_base64") or item.get("content_base64")
        if raw_b64:
            try:
                decoded = base64.b64decode(raw_b64).decode("utf-8")
                texts.append(decoded)
            except Exception:
                pass

    if texts:
        return texts

    # Source 3: direct clinical notes list
    for note in payload.get("clinical_notes") or []:
        if isinstance(note, str) and note.strip():
            texts.append(note.strip())

    # Source 4: single text field
    text_field = payload.get("text")
    if isinstance(text_field, str) and text_field.strip():
        texts.append(text_field.strip())

    # Source 5: analysis_artifacts preview as fallback
    if not texts:
        artifacts = payload.get("analysis_artifacts") or {}
        for key, artifact in artifacts.items():
            if isinstance(artifact, dict):
                preview = artifact.get("preview")
                if isinstance(preview, str) and preview.strip():
                    texts.append(preview.strip())

    return texts


def run_inference(clinical_texts: list[str], model_path: str) -> list[dict]:
    """Run dementia prediction on one or more clinical notes."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    llm = LLM(
        model=model_path,
        tensor_parallel_size=TENSOR_PARALLEL,
        gpu_memory_utilization=GPU_UTIL,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        seed=42,
    )

    sampling = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        top_p=1.0,
        logprobs=20,
        stop=["</answer>"],
    )

    prompts = [build_prompt(text, tokenizer) for text in clinical_texts]
    raw_outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)

    results = []
    for i, output in enumerate(raw_outputs):
        generated_text = output.outputs[0].text if output.outputs else ""
        pred_label = extract_label(generated_text)
        prob_score = get_prob_score(output, tokenizer)

        results.append({
            "patient_index": i,
            "prediction": int(pred_label) if pred_label else None,
            "prediction_label": "likely dementia" if pred_label == "1" else (
                "unlikely dementia" if pred_label == "0" else "uncertain"
            ),
            "probability_score": round(prob_score, 4),
            "reasoning": generated_text.strip(),
        })

    return results


def _clean_reasoning(raw: str) -> str:
    """Extract readable reasoning from model output, stripping tags."""
    text = raw.strip()
    # Extract content between <think> and </think>
    m = re.search(r"<think>\s*(.*?)\s*(?:</think>|$)", text, flags=re.DOTALL)
    if m:
        text = m.group(1).strip()
    # Remove any remaining tags
    text = re.sub(r"</?(?:think|answer)>", "", text).strip()
    # Remove \boxed{...}
    text = re.sub(r"\\boxed\{[^}]*\}", "", text).strip()
    return text


def format_markdown_summary(results: list[dict]) -> str:
    """Create a human-readable markdown summary of predictions."""
    lines = ["## Dementia Prediction Results\n"]

    for r in results:
        idx = r["patient_index"]
        pred = r["prediction"]
        label = r["prediction_label"]
        prob = r["probability_score"]
        reasoning = _clean_reasoning(r.get("reasoning", ""))
        icon = "\u26a0\ufe0f" if pred == 1 else "\u2705"

        lines.append(f"### Patient #{idx + 1}")
        lines.append(f"- **Prediction**: {icon} **{label}** (class={pred})")
        lines.append(f"- **Probability of dementia**: **{prob:.1%}**")
        lines.append("")
        if reasoning:
            lines.append("#### Model Reasoning")
            lines.append("")
            lines.append(reasoning)
            lines.append("")

    lines.append("---")
    lines.append("*Model: Dementia-R1 (Qwen2 fine-tuned) | This prediction is for research purposes only and should not replace clinical judgment.*")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dementia Prediction Tool")
    parser.add_argument("--input", required=True, help="Path to input JSON")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument("--model", default=None, help="Override model checkpoint path")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    execution_context = payload.get("execution_context") or {}

    # Debug: dump payload keys to stderr for diagnosing integration issues
    import sys
    print(f"[dementia_prediction_tool] payload keys: {list(payload.keys())}", file=sys.stderr)
    # Save full payload for debugging
    debug_path = Path("/tmp/dementia_debug_payload.json")
    debug_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[dementia_prediction_tool] payload saved to {debug_path}", file=sys.stderr)

    model_path = args.model or DEFAULT_MODEL_PATH
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            f"Set DEMENTIA_MODEL_PATH env var or pass --model."
        )

    clinical_texts = extract_clinical_text(payload)
    if not clinical_texts:
        raise ValueError(
            "No clinical notes found in payload. "
            "Provide text files, clinical_notes list, or text field."
        )

    predictions = run_inference(clinical_texts, model_path)
    summary = format_markdown_summary(predictions)

    result = {
        "summary": summary,
        "artifacts": {
            "dementia_prediction": {
                "predictions": predictions,
                "num_patients": len(predictions),
            },
            "model_info": {
                "name": "Dementia-R1",
                "checkpoint": str(model_path),
                "architecture": "Qwen2ForCausalLM",
            },
        },
        "provenance": {
            "model": "Dementia-R1 (Qwen2 fine-tuned)",
            "tool": "dementia_prediction_tool",
        },
        "used_tools": ["dementia_prediction_tool"],
        "execution_context": execution_context,
    }

    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
