#!/usr/bin/env python3
"""
Dementia Prediction API Server
================================
muzi 서버에서 상시 실행되는 독립 API 서버.
교수님 Mac의 ChatClinic에서 원격으로 호출할 수 있습니다.

실행:
  CUDA_VISIBLE_DEVICES=0 python api_server.py

환경변수:
  DEMENTIA_API_PORT    (기본: 8020)
  DEMENTIA_MODEL_PATH  (기본: 자동 탐색)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# GPU 핀 설정 (다른 import 전에)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DEMENTIA_GPU_DEVICE", "0"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 같은 디렉토리의 run.py에서 핵심 로직 재사용
sys.path.insert(0, str(Path(__file__).parent))
from run import (
    DEFAULT_MODEL_PATH,
    MAX_TOKENS,
    MAX_MODEL_LEN,
    TENSOR_PARALLEL,
    GPU_UTIL,
    extract_label,
    get_prob_score,
    build_prompt,
    _clean_reasoning,
    format_markdown_summary,
)

app = FastAPI(title="Dementia Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 모델을 서버 시작 시 한 번만 로드 (매 요청마다 로드하지 않음)
# ---------------------------------------------------------------------------
_llm = None
_tokenizer = None
_sampling = None


def _load_model():
    global _llm, _tokenizer, _sampling
    if _llm is not None:
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_path = os.environ.get("DEMENTIA_MODEL_PATH", DEFAULT_MODEL_PATH)
    if not Path(model_path).exists():
        raise RuntimeError(f"Model not found: {model_path}")

    print(f"[API] Loading model from {model_path} ...")
    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _llm = LLM(
        model=model_path,
        tensor_parallel_size=TENSOR_PARALLEL,
        gpu_memory_utilization=GPU_UTIL,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        seed=42,
    )
    _sampling = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        top_p=1.0,
        logprobs=20,
        stop=["</answer>"],
    )
    print("[API] Model loaded successfully.")


# ---------------------------------------------------------------------------
# API 모델
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    clinical_notes: list[str]


class PredictionItem(BaseModel):
    patient_index: int
    prediction: int | None
    prediction_label: str
    probability_score: float
    reasoning: str


class PredictResponse(BaseModel):
    predictions: list[PredictionItem]
    summary: str
    model: str = "Dementia-R1"
    elapsed_sec: float


# ---------------------------------------------------------------------------
# 엔드포인트
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _llm is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.clinical_notes:
        raise HTTPException(status_code=400, detail="clinical_notes list is empty")

    _load_model()

    start = time.time()

    prompts = [build_prompt(text, _tokenizer) for text in req.clinical_notes]
    raw_outputs = _llm.generate(prompts, sampling_params=_sampling, use_tqdm=False)

    results = []
    for i, output in enumerate(raw_outputs):
        generated_text = output.outputs[0].text if output.outputs else ""
        pred_label = extract_label(generated_text)
        prob_score = get_prob_score(output, _tokenizer)

        results.append(PredictionItem(
            patient_index=i,
            prediction=int(pred_label) if pred_label else None,
            prediction_label="likely dementia" if pred_label == "1" else (
                "unlikely dementia" if pred_label == "0" else "uncertain"
            ),
            probability_score=round(prob_score, 4),
            reasoning=_clean_reasoning(generated_text),
        ))

    summary = format_markdown_summary([r.model_dump() for r in results])
    elapsed = round(time.time() - start, 1)

    return PredictResponse(
        predictions=results,
        summary=summary,
        elapsed_sec=elapsed,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("DEMENTIA_API_PORT", "8020"))

    # 서버 시작 전에 모델 미리 로드
    _load_model()

    print(f"[API] Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
