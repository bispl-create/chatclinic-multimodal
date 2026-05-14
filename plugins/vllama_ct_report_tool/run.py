from __future__ import annotations

from app.services.plugin_runtime import run_plugin_cli
from plugins.vllama_ct_report_tool.logic import execute


if __name__ == "__main__":
    run_plugin_cli(execute)
