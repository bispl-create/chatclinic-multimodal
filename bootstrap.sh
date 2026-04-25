#!/usr/bin/env bash
# One-shot setup for ChatClinic class demo.
# Runs every step from SETUP.md §2-§7. Safe to re-run (idempotent).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "=== [1/7] Ensure uv + gdown + huggingface_hub are installed ==="
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v gdown            >/dev/null 2>&1 || uv tool install gdown
command -v huggingface-cli  >/dev/null 2>&1 || uv tool install huggingface_hub

echo "=== [2/7] Clone the 4 external backends (skip if already present) ==="
declare -A BACKENDS=(
    [CoreDiff]=https://github.com/qgao21/CoreDiff.git
    [Fast-DDPM]=https://github.com/mirthAI/Fast-DDPM.git
    [SNRAware]=https://github.com/microsoft/SNRAware.git
    [SharpXR]=https://github.com/ileri-oluwa-kiiye/SharpXR.git
)
mkdir -p external_backends
for name in "${!BACKENDS[@]}"; do
    dest="external_backends/$name"
    if [ -d "$dest/.git" ]; then
        echo "  $name already cloned"
    else
        rm -rf "$dest"
        git clone "${BACKENDS[$name]}" "$dest"
    fi
done

echo "=== [3/7] Install SNRAware's isolated venv (also used by the whole app) ==="
(cd external_backends/SNRAware && uv sync)

echo "=== [4/7] Install ChatClinic backend deps into that same venv ==="
uv pip install --python external_backends/SNRAware/.venv \
    'fastapi>=0.115,<1.0' 'uvicorn>=0.30' 'python-multipart' \
    'pydantic>=2.8,<3.0' 'pysam>=0.23' 'openpyxl' 'Pillow>=10.0' \
    'nibabel>=5.2' pydicom requests torchxrayvision

echo "=== [5/7] Download model weights (skip files that already exist) ==="
mkdir -p external_weights/{CoreDiff,FastDDPM,SNRAware,SharpXR}
if [ ! -s external_weights/CoreDiff/checkpoint/ema_model-150000 ]; then
    gdown --folder "https://drive.google.com/drive/folders/1rGb34H_6ktP79vMYYJOLSoCE3579TDZ5" \
          -O external_weights/CoreDiff
else
    echo "  CoreDiff weights already present"
fi
for f in ckpt_LDFDCT.pth ckpt_PMUB.pth ckpt_BRATS.pth; do
    if [ ! -s "external_weights/FastDDPM/$f" ]; then
        huggingface-cli download SebastianJiang/FastDDPM "$f" \
            --local-dir external_weights/FastDDPM
    fi
done
if [ ! -d external_weights/SNRAware/small ]; then
    huggingface-cli download microsoft/SNRAware \
        --local-dir external_weights/SNRAware
else
    echo "  SNRAware weights already present"
fi
if [ ! -s external_weights/SharpXR/sharpxr_best.pth ]; then
    gdown "https://drive.google.com/uc?id=1ME1fhGse95E5FZ7qwqvr0ykM1XH3h4if" \
        -O external_weights/SharpXR/sharpxr_best.pth
else
    echo "  SharpXR weights already present"
fi

echo "=== [6/7] Install the Next.js frontend ==="
(cd chatclinic-multimodal && npm install --workspace webapp)

echo "=== [7/7] Create .env if missing ==="
if [ ! -f chatclinic-multimodal/.env ]; then
    cp chatclinic-multimodal/sample.env chatclinic-multimodal/.env
    echo
    echo "    !! Edit chatclinic-multimodal/.env and set OPENAI_API_KEY=sk-... before running the app."
    echo
fi

echo
echo "Bootstrap complete. Next: bash start_demo.sh"
