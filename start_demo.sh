#!/usr/bin/env bash
# Start ChatClinic backend + frontend + pre-warm the GPU.
# Leaves both servers running in the background. Stops with `pkill -f uvicorn; pkill -f "next dev"`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

if [ ! -d chatclinic-multimodal ] || [ ! -d external_backends/SNRAware/.venv ]; then
    echo "Run bash bootstrap.sh first." >&2
    exit 1
fi

if [ ! -d demo_images/noisy ] || [ -z "$(ls -A demo_images/noisy 2>/dev/null)" ]; then
    echo "=== Generating demo images ==="
    (cd chatclinic-multimodal && uv run --project ../external_backends/SNRAware python3 \
        -m plugins.medical_restoration_common.build_demo_inputs)
fi

echo "=== Launching backend (port 8001, GPU 0) ==="
(cd chatclinic-multimodal && \
    CUDA_VISIBLE_DEVICES=0 uv run --project ../external_backends/SNRAware \
        python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --log-level info \
        > /tmp/chatclinic_backend.log 2>&1 &)

echo "=== Launching frontend (port 3000) ==="
(cd chatclinic-multimodal && npm --workspace webapp run dev \
    > /tmp/chatclinic_frontend.log 2>&1 &)

echo "=== Waiting for servers ==="
until curl -s http://127.0.0.1:8001/api/v1/tools > /dev/null 2>&1; do sleep 2; done
echo "  backend ready"
until curl -s http://127.0.0.1:3000 > /dev/null 2>&1; do sleep 2; done
echo "  frontend ready"

echo "=== Pre-warming GPU ==="
(cd chatclinic-multimodal && uv run --project ../external_backends/SNRAware python3 \
    -m plugins.medical_restoration_common.prewarm \
    --backend http://127.0.0.1:8001 \
    --demo-dir ../demo_images)

echo
echo "Demo is ready."
echo "  Frontend: http://127.0.0.1:3000"
echo "  Backend:  http://127.0.0.1:8001"
echo "  Logs:     /tmp/chatclinic_backend.log and /tmp/chatclinic_frontend.log"
echo "  Watch GPU: watch -n 1 nvidia-smi"
echo "  Stop all: pkill -f 'uvicorn app.main' ; pkill -f 'next dev'"
