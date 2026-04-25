from __future__ import annotations

import os
from pathlib import Path

# Repo root is three levels up: plugins/medical_restoration_common/paths.py
APP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = APP_ROOT.parent  # /home/ttran/kaist_projects/chatclinic-class

EXTERNAL_BACKENDS = Path(os.environ.get("CHATCLINIC_BACKENDS_DIR", PROJECT_ROOT / "external_backends"))
EXTERNAL_WEIGHTS = Path(os.environ.get("CHATCLINIC_WEIGHTS_DIR", PROJECT_ROOT / "external_weights"))

SNRAWARE_BACKEND = EXTERNAL_BACKENDS / "SNRAware"
COREDIFF_BACKEND = EXTERNAL_BACKENDS / "CoreDiff"
FASTDDPM_BACKEND = EXTERNAL_BACKENDS / "Fast-DDPM"

SNRAWARE_WEIGHTS = EXTERNAL_WEIGHTS / "SNRAware"
FASTDDPM_WEIGHTS = EXTERNAL_WEIGHTS / "FastDDPM"
COREDIFF_WEIGHTS = EXTERNAL_WEIGHTS / "CoreDiff"

RESTORATION_CACHE = Path(os.environ.get("CHATCLINIC_RESTORATION_CACHE", PROJECT_ROOT / ".cache" / "restoration"))
RESTORATION_CACHE.mkdir(parents=True, exist_ok=True)


def backend_status() -> dict[str, dict[str, object]]:
    return {
        "snraware": {
            "backend_dir": str(SNRAWARE_BACKEND),
            "weights_dir": str(SNRAWARE_WEIGHTS),
            "backend_exists": SNRAWARE_BACKEND.exists(),
            "weights_exist": SNRAWARE_WEIGHTS.exists() and any(SNRAWARE_WEIGHTS.iterdir()) if SNRAWARE_WEIGHTS.exists() else False,
        },
        "corediff": {
            "backend_dir": str(COREDIFF_BACKEND),
            "weights_dir": str(COREDIFF_WEIGHTS),
            "backend_exists": COREDIFF_BACKEND.exists(),
            "weights_exist": COREDIFF_WEIGHTS.exists(),
        },
        "fastddpm": {
            "backend_dir": str(FASTDDPM_BACKEND),
            "weights_dir": str(FASTDDPM_WEIGHTS),
            "backend_exists": FASTDDPM_BACKEND.exists(),
            "weights_exist": FASTDDPM_WEIGHTS.exists(),
        },
    }
