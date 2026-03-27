from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.models import ToolInfo
from app.services.plugin_runtime import load_plugin_execute, serialize_plugin_result


ROOT_DIR = Path(__file__).resolve().parents[2]
PLUGINS_DIR = ROOT_DIR / "plugins"


@lru_cache(maxsize=1)
def load_tool_manifests() -> list[dict[str, object]]:
    manifests: list[dict[str, object]] = []
    if not PLUGINS_DIR.exists():
        return manifests
    for manifest_path in sorted(PLUGINS_DIR.glob("*/tool.json")):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            manifests.append(payload)
    return manifests


def tool_aliases(manifest: dict[str, object]) -> list[str]:
    aliases: list[str] = []

    def add_alias(value: str) -> None:
        text = value.strip().lower()
        if text and text not in aliases:
            aliases.append(text)

    name = str(manifest.get("name") or "").strip().lower()
    if name:
        simplified = re.sub(r"^(gatk_|bcftools_)", "", name)
        while True:
            next_value = re.sub(r"_(execution|vcf|tool)$", "", simplified)
            if next_value == simplified:
                break
            simplified = next_value
        simplified = re.sub(r"_+", "_", simplified).strip("_")
        if simplified:
            add_alias(simplified)
            add_alias(simplified.replace("_", ""))
            add_alias(simplified.replace("_", "-"))
        add_alias(name)
    routing = manifest.get("routing")
    if isinstance(routing, dict):
        for keyword in routing.get("trigger_keywords", []):
            text = str(keyword).strip().lower()
            if text and re.fullmatch(r"[a-z0-9_-]+", text):
                add_alias(text)
    return aliases


def manifest_for_tool_name(tool_name: str | None) -> dict[str, object] | None:
    if not tool_name:
        return None
    normalized = str(tool_name).strip()
    for manifest in load_tool_manifests():
        if str(manifest.get("name") or "").strip() == normalized:
            return manifest
    return None


def manifest_for_alias(alias: str | None) -> dict[str, object] | None:
    if not alias:
        return None
    lowered = str(alias).strip().lower()
    for manifest in load_tool_manifests():
        if lowered in tool_aliases(manifest):
            return manifest
    return None


def infer_tool_source_types(manifest: dict[str, object]) -> list[str]:
    workflow_binding = manifest.get("workflow_binding")
    source_types: set[str] = set()
    if isinstance(workflow_binding, dict):
        source_type = str(workflow_binding.get("source_type") or "").strip().lower()
        if source_type:
            source_types.add(source_type)

    orchestration = manifest.get("orchestration")
    consumes = orchestration.get("consumes") if isinstance(orchestration, dict) else []
    if isinstance(consumes, list):
        lowered = [str(item).strip().lower() for item in consumes]
        if "vcf_path" in lowered:
            source_types.add("vcf")
        if "alignment_file" in lowered or "raw_sequence_file" in lowered:
            source_types.add("raw_qc")
        if "summary_stats_path" in lowered:
            source_types.add("summary_stats")
    return sorted(source_types)


def infer_tool_result_kind(manifest: dict[str, object]) -> str | None:
    direct_chat = manifest.get("direct_chat")
    if isinstance(direct_chat, dict):
        result_kind = str(direct_chat.get("result_kind") or "").strip()
        if result_kind:
            return result_kind
    routing = manifest.get("routing")
    if isinstance(routing, dict):
        result_slot = str(routing.get("result_slot") or "").strip()
        if result_slot:
            return result_slot
    workflow_binding = manifest.get("workflow_binding")
    if isinstance(workflow_binding, dict):
        result_path = str(workflow_binding.get("result_path") or "").strip()
        if result_path:
            return result_path
    return None


def tool_direct_chat_metadata(manifest: dict[str, object]) -> dict[str, Any]:
    direct_chat = manifest.get("direct_chat")
    if not isinstance(direct_chat, dict):
        return {}
    payload = dict(direct_chat)
    payload.setdefault("source_type", next(iter(infer_tool_source_types(manifest)), ""))
    payload.setdefault("result_kind", infer_tool_result_kind(manifest))
    payload.setdefault("aliases", tool_aliases(manifest))
    payload.setdefault("name", str(manifest.get("name") or "").strip())
    payload.setdefault("help_supported", isinstance(manifest.get("help"), dict))
    return payload


def tool_chat_metadata(manifest: dict[str, object]) -> dict[str, Any]:
    direct_chat = tool_direct_chat_metadata(manifest)
    return {
        "name": str(manifest.get("name") or "").strip(),
        "aliases": tool_aliases(manifest),
        "source_types": infer_tool_source_types(manifest),
        "result_kind": infer_tool_result_kind(manifest),
        "help_supported": isinstance(manifest.get("help"), dict),
        "direct_preanalysis_supported": isinstance(manifest.get("routing"), dict),
        "direct_chat": direct_chat,
    }


def discover_tools() -> list[ToolInfo]:
    tools: list[ToolInfo] = []
    for payload in load_tool_manifests():
        tools.append(
            ToolInfo(
                name=str(payload.get("name", "tool")),
                description=str(payload.get("description", "")),
                task=str(payload.get("task", "unknown")),
                modality=str(payload.get("modality", "genomics")),
                approval_required=bool(payload.get("approval_required", False)),
                source=str(payload.get("source", "plugin")),
            )
        )
    return tools


def run_tool(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    manifest = manifest_for_tool_name(tool_name)
    if manifest is None:
        raise FileNotFoundError(f"Tool manifest not found for {tool_name}")

    entrypoint = str(manifest.get("entrypoint") or "").strip()
    if entrypoint:
        execute = load_plugin_execute(entrypoint)
        result = serialize_plugin_result(execute(payload))
        if not isinstance(result, dict):
            raise RuntimeError(f"Tool {tool_name} returned a non-dict result for entrypoint execution.")
        return result

    tool_dir = _find_tool_dir(tool_name)
    run_path = tool_dir / "run.py"
    if not run_path.exists():
        raise FileNotFoundError(f"Tool runner not found for {tool_name}: {run_path}")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as input_file:
        json.dump(payload, input_file, ensure_ascii=False)
        input_path = input_file.name

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as output_file:
        output_path = output_file.name

    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    pythonpath_parts = [str(ROOT_DIR)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    command = [sys.executable, str(run_path), "--input", input_path, "--output", output_path]
    completed = subprocess.run(
        command,
        cwd=str(ROOT_DIR),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Tool {tool_name} failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )

    return json.loads(Path(output_path).read_text(encoding="utf-8"))


def _find_tool_dir(tool_name: str) -> Path:
    for manifest_path in PLUGINS_DIR.glob("*/tool.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(payload.get("name")) == tool_name:
            return manifest_path.parent
    raise FileNotFoundError(f"Tool manifest not found for {tool_name}")
