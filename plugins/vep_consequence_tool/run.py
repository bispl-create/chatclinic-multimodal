from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.plugin_runtime import run_plugin_cli  # noqa: E402
from plugins.vep_consequence_tool.logic import execute  # noqa: E402


def main() -> None:
    run_plugin_cli(execute)


if __name__ == "__main__":
    main()
