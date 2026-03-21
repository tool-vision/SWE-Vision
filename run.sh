#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Local vLLM baseline ────────────────────────────────────────
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"

# ─── Run agent with an image question ───────────────────────────
IMAGE_PATH="${1:-./assets/test_image.png}"
QUERY="${2:-What is the gap between GPT5.2 and 6-year-olds from the chart?}"

python -m swe_vision.cli \
    --image "$IMAGE_PATH" \
    --model "$MODEL_NAME" \
    --no-reasoning \
    "$QUERY"
