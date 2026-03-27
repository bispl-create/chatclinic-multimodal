from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Callable


def serialize_plugin_result(result: Any) -> Any:
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return result


def load_plugin_execute(entrypoint: str) -> Callable[[dict[str, Any]], Any]:
    module_name, separator, function_name = entrypoint.partition(":")
    if not separator or not module_name.strip() or not function_name.strip():
        raise ValueError(f"Invalid plugin entrypoint: {entrypoint}")
    module = importlib.import_module(module_name.strip())
    execute = getattr(module, function_name.strip(), None)
    if not callable(execute):
        raise ValueError(f"Plugin entrypoint is not callable: {entrypoint}")
    return execute


def run_plugin_cli(execute: Callable[[dict[str, Any]], Any]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    serialized = serialize_plugin_result(execute(payload))
    Path(args.output).write_text(
        json.dumps(serialized, ensure_ascii=False),
        encoding="utf-8",
    )
