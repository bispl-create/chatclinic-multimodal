#!/usr/bin/env bash
# Download external restoration backends into this plugin and checkpoints into
# the repo-level ckpt_and_file directory. Safe to re-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKENDS_DIR="$SCRIPT_DIR/external_backends"
WEIGHTS_DIR="$REPO_ROOT/ckpt_and_file/medical_restoration_common"

echo "=== Ensure download helpers are available ==="
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v gdown >/dev/null 2>&1 || uv tool install gdown
if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
    uv tool install huggingface_hub
fi
HF_DOWNLOAD=(hf download)
if ! command -v hf >/dev/null 2>&1; then
    HF_DOWNLOAD=(huggingface-cli download)
fi

echo "=== Clone external restoration backends ==="
declare -A BACKENDS=(
    [CoreDiff]=https://github.com/qgao21/CoreDiff.git
    [Fast-DDPM]=https://github.com/mirthAI/Fast-DDPM.git
    [SNRAware]=https://github.com/microsoft/SNRAware.git
    [SharpXR]=https://github.com/ileri-oluwa-kiiye/SharpXR.git
)
mkdir -p "$BACKENDS_DIR"
for name in "${!BACKENDS[@]}"; do
    dest="$BACKENDS_DIR/$name"
    if [ -d "$dest/.git" ]; then
        echo "  $name already cloned"
    else
        rm -rf "$dest"
        git clone "${BACKENDS[$name]}" "$dest"
    fi
done

echo "=== Prepare SNRAware backend environment ==="
SNRAWARE_PYTHON="${SNRAWARE_PYTHON:-3.12}"
if command -v uv >/dev/null 2>&1; then
    (cd "$BACKENDS_DIR/SNRAware" && uv sync --python "$SNRAWARE_PYTHON")
else
    echo "  uv not found; SNRAware will prepare its environment on first run"
fi

echo "=== Download restoration checkpoints ==="
mkdir -p "$WEIGHTS_DIR"/{CoreDiff,FastDDPM,SNRAware,SharpXR}

if [ ! -s "$WEIGHTS_DIR/CoreDiff/checkpoint/ema_model-150000" ]; then
    gdown --folder "https://drive.google.com/drive/folders/1rGb34H_6ktP79vMYYJOLSoCE3579TDZ5" \
        -O "$WEIGHTS_DIR/CoreDiff"
else
    echo "  CoreDiff weights already present"
fi

for f in ckpt_LDFDCT.pth ckpt_PMUB.pth ckpt_BRATS.pth; do
    if [ ! -s "$WEIGHTS_DIR/FastDDPM/$f" ]; then
        "${HF_DOWNLOAD[@]}" SebastianJiang/FastDDPM "$f" \
            --local-dir "$WEIGHTS_DIR/FastDDPM"
    else
        echo "  FastDDPM $f already present"
    fi
done

if [ ! -d "$WEIGHTS_DIR/SNRAware/small" ]; then
    "${HF_DOWNLOAD[@]}" microsoft/SNRAware \
        --local-dir "$WEIGHTS_DIR/SNRAware"
else
    echo "  SNRAware weights already present"
fi

if [ ! -s "$WEIGHTS_DIR/SharpXR/sharpxr_best.pth" ]; then
    gdown "https://drive.google.com/uc?id=1ME1fhGse95E5FZ7qwqvr0ykM1XH3h4if" \
        -O "$WEIGHTS_DIR/SharpXR/sharpxr_best.pth"
else
    echo "  SharpXR weights already present"
fi

echo
echo "Restoration assets are ready:"
echo "  Backends: $BACKENDS_DIR"
echo "  Weights:  $WEIGHTS_DIR"
