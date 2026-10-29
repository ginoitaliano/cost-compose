#!/usr/bin/env bash
# Serve Llama-3-8B-Instruct with an OpenAI-compatible API on the Studio GPU.
# Then: export HOSTED_VLLM_API_BASE=http://localhost:8000/v1
set -euo pipefail

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
PORT="${PORT:-8000}"

pip install --quiet "vllm>=0.6"
exec vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json
