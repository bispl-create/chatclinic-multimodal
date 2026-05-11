"""Parkinson's disease Enhanced RAG medication planning plugin.

Wraps the --enhanced_rag flow from parkinson_plan_prediction/main.py for
single-patient use inside the ChatClinic workspace.

Flow (non-MIMIC, per README_ENHANCED_RAG.md):
  Step 1  – Focus keyword extraction (LLM)
  Step 2  – Per-focus FAISS retrieval
  Step 3  – Per-focus tendency analysis (LLM)
  Step 4  – Initial prescription (LLM, history-first)
  Step 5  – Delta Verifier (LLM, anchored on active history)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: inject parkinson venv site-packages + parkinson source dir
# so that recommend_plan / prompt (and their deps) are importable.
# ---------------------------------------------------------------------------
_DEFAULT_PARKINSON_ROOT = "/mnt/data1/hyunmin/parkinson_plan_prediction"


def _parkinson_root() -> str:
    return os.environ.get("PARKINSON_ROOT", _DEFAULT_PARKINSON_ROOT)


def _parkinson_source_dir() -> str:
    root = _parkinson_root()
    return os.environ.get("PARKINSON_SOURCE_DIR", root)


def _parkinson_venv_site() -> str:
    root = _parkinson_root()
    return os.environ.get(
        "PARKINSON_VENV_SITE",
        os.path.join(root, ".venv_parkinson", "lib", "python3.10", "site-packages"),
    )


def _vector_db_path() -> str:
    root = _parkinson_root()
    return os.environ.get(
        "PARKINSON_VECTOR_DB_PATH",
        os.path.join(root, "vector_db", "parkinson_openr1_train_replaced_soap_faiss_index"),
    )


def _bm25_metadata_path() -> str:
    root = _parkinson_root()
    return os.environ.get(
        "PARKINSON_BM25_METADATA",
        os.path.join(root, "data", "parkinson_openr1_train_replaced_soap.json"),
    )


def _medicine_info_path() -> str:
    root = _parkinson_root()
    return os.environ.get(
        "PARKINSON_MEDICINE_INFO",
        os.path.join(root, "data", "medicine_info_gpt40mini_updated4.json"),
    )


def _load_parkinson_modules():
    source_dir = _parkinson_source_dir()
    venv_site = _parkinson_venv_site()
    for path in (venv_site, source_dir):
        if Path(path).exists() and path not in sys.path:
            sys.path.insert(0, path)

    missing = [
        ("PARKINSON_SOURCE_DIR", source_dir, "recommend_plan.py / prompt.py"),
        ("PARKINSON_VECTOR_DB_PATH", _vector_db_path(), "FAISS vector DB"),
        ("PARKINSON_BM25_METADATA", _bm25_metadata_path(), "BM25 metadata JSON"),
        ("PARKINSON_MEDICINE_INFO", _medicine_info_path(), "medicine info JSON"),
    ]
    missing = [f"{env_name}={path} ({label})" for env_name, path, label in missing if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Parkinson plan dependency paths are missing. Set PARKINSON_ROOT or the specific "
            "PARKINSON_* path variables. Missing: " + "; ".join(missing)
        )

    try:
        import recommend_plan as rp
        from recommend_plan import rag_w_similar_patients_plan, preprocess_user_input
        from prompt import (
            LLM_recommend_plan_answer_with_retrieved_plans,
            call_LLM_doctor_summary,
            format_reflective_qa_for_doctor_summary,
            parse_prescription_to_list,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import Parkinson source modules. Ensure PARKINSON_SOURCE_DIR points to the "
            "directory containing recommend_plan.py and prompt.py, and install that project's dependencies."
        ) from exc

    rp.BM25_METADATA = _bm25_metadata_path()
    rp.MEDICINE_INFORMATION = _medicine_info_path()
    return (
        rag_w_similar_patients_plan,
        preprocess_user_input,
        LLM_recommend_plan_answer_with_retrieved_plans,
        call_LLM_doctor_summary,
        format_reflective_qa_for_doctor_summary,
        parse_prescription_to_list,
    )

# ---------------------------------------------------------------------------
# Lazy LLM singleton
# ---------------------------------------------------------------------------
_llm = None


def _get_llm():
    global _llm
    if _llm is not None:
        return _llm

    model = os.environ.get("PARKINSON_LLM_MODEL", "qwen3:8b")
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11437")

    from langchain_ollama import OllamaLLM

    _llm = OllamaLLM(
        model=model,
        base_url=f"http://{host}",
        temperature=float(os.environ.get("PARKINSON_LLM_TEMP", "0.2")),
    )
    return _llm


# ---------------------------------------------------------------------------
# Plugin entrypoint
# ---------------------------------------------------------------------------

def execute(payload: dict) -> dict:
    """Run Enhanced RAG prescription for a single Parkinson's patient.

    Expected payload keys:
      subjective            (str)  – SOAP Subjective section
      objective             (str)  – SOAP Objective section
      assessment            (str)  – SOAP Assessment section
      recent_visit_history  (list) – list of visit dicts; each has 'prescription' (list[str])
      patient_id            (str)  – patient identifier (used to exclude self from RAG)
      threshold             (float)– FAISS similarity threshold  (default 0.7)
      retrieve_patients     (int)  – top-k per focus query       (default 4)
    """
    subjective = payload.get("subjective") or ""
    objective = payload.get("objective") or ""
    assessment = payload.get("assessment") or ""
    patient_id = payload.get("patient_id", "unknown")
    recent_visit_history = payload.get("recent_visit_history") or []
    threshold = float(payload.get("threshold", 0.7))
    retrieve_patients = int(payload.get("retrieve_patients", 4))

    (
        rag_w_similar_patients_plan,
        preprocess_user_input,
        LLM_recommend_plan_answer_with_retrieved_plans,
        call_LLM_doctor_summary,
        format_reflective_qa_for_doctor_summary,
        parse_prescription_to_list,
    ) = _load_parkinson_modules()
    llm = _get_llm()

    input_full_text = preprocess_user_input(subjective, objective, assessment)

    # i_note dict expected by rag_w_similar_patients_plan
    i_note = {
        "subjective": subjective or None,
        "objective": objective or None,
        "assessment": assessment or None,
    }

    # Active history = prescription from the most recent visit
    active_history: list[str] = []
    if recent_visit_history:
        last_visit = recent_visit_history[-1]
        active_history = last_visit.get("prescription") or []

    is_first_visit = not bool(active_history)

    # ------------------------------------------------------------------
    # Steps 1-3: RAG retrieval + per-focus tendency analysis
    # ------------------------------------------------------------------
    (
        _plan_summary_dict,
        plan_summary_text,
        _rag_time,
        focus_meta,
    ) = rag_w_similar_patients_plan(
        input_full_text,
        i_note,
        patient_id,
        llm,
        threshold,
        _vector_db_path(),
        enhanced_rag=True,
        retrieve_patients=retrieve_patients,
        active_history=active_history,
        recent_visit_history=recent_visit_history,
    )

    focus_areas: dict = {}
    rag_tendency_by_focus: list = []
    if isinstance(focus_meta, dict):
        focus_areas = focus_meta.get("focus_areas") or {}
        rag_tendency_by_focus = focus_meta.get("rag_tendency_by_focus") or []

    # ------------------------------------------------------------------
    # Steps 4-5: Initial prescription + Delta Verifier
    # ------------------------------------------------------------------
    (
        draft_plan,
        verified_output,
        _active_history_list,
        _clinical_status,
        _gen_time,
        _ver_time,
        _h_time,
        _prof_time,
        _aud_time,
    ) = LLM_recommend_plan_answer_with_retrieved_plans(
        input_full_text,
        plan_summary_text,
        llm,
        subjective=subjective,
        objective=objective,
        assessment=assessment,
        enhanced_rag=True,
        recent_visit_history=recent_visit_history,
        is_first_visit=is_first_visit,
        rag_tendency_by_focus=rag_tendency_by_focus,
        extracted_focus_keywords=focus_areas,
        enhanced_rag_keyword_mode=True,
    )

    # ------------------------------------------------------------------
    # Parse final prescription and audit log
    # ------------------------------------------------------------------
    final_prescription: list[str] = []
    audit_log: list = []

    if isinstance(verified_output, dict):
        raw_fp = verified_output.get("final_prescription") or verified_output.get("final_prescription_list") or []
        final_prescription = [str(d).strip() for d in raw_fp if d and str(d).strip()]
        raw_al = verified_output.get("audit_log")
        audit_log = raw_al if isinstance(raw_al, list) else []
    else:
        parsed = parse_prescription_to_list(verified_output) if isinstance(verified_output, str) else []
        final_prescription = [str(d).strip() for d in parsed if d and str(d).strip()]

    # Fallback: if verifier returned nothing, use draft
    if not final_prescription and draft_plan:
        parsed_draft = parse_prescription_to_list(draft_plan) if isinstance(draft_plan, str) else []
        final_prescription = [str(d).strip() for d in parsed_draft if d and str(d).strip()]

    patient_state = f"S: {subjective or 'None'} | O: {objective or 'None'} | A: {assessment or 'None'}"
    clinical_evidence = rag_tendency_by_focus
    if isinstance(verified_output, dict):
        reflective_question = verified_output.get("reflective_question")
        reflective_answer = verified_output.get("reflective_answer")
        if reflective_question or reflective_answer:
            reflective_evidence = format_reflective_qa_for_doctor_summary(reflective_question, reflective_answer)
            if reflective_evidence:
                clinical_evidence = reflective_evidence

    doctor_summary = call_LLM_doctor_summary(
        patient_state=patient_state,
        initial_prescription=draft_plan,
        rag_tendency_by_focus=clinical_evidence,
        audit_log=audit_log,
        final_answer_list={"final_prescription": final_prescription},
        llm=llm,
    )

    return {
        "patient_id": patient_id,
        "draft_plan": draft_plan if isinstance(draft_plan, str) else str(draft_plan),
        "final_prescription": final_prescription,
        "focus_areas": focus_areas,
        "rag_tendency_by_focus": rag_tendency_by_focus,
        "audit_log": audit_log,
        "doctor_summary": doctor_summary,
        "summary": (
            f"Enhanced RAG plan for patient {patient_id}: "
            f"{len(final_prescription)} drug(s) in final prescription."
        ),
    }
