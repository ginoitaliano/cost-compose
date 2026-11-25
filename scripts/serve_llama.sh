#!/usr/bin/env bash
# Serve Llama-3.1-8B-Instruct with an OpenAI-compatible API on the Studio GPU.
# Then: export HOSTED_VLLM_API_BASE=http://localhost:8000/v1
#
# Llama 3.1 is gated: accept Meta's license on the model page with the Hugging Face account
# whose token is in HF_TOKEN. The model id must match the ladder entry in configs/base.yaml.
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
HOST="${HOST:-127.0.0.1}"            # local only; vLLM has no auth
PORT="${PORT:-8000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # an 8B model needs one GPU
# Llama 3.1's native context length. The other models on the ladder accept 128k tokens or more,
# so a shorter limit would fail Llama on long tool histories for a reason that is ours, not the
# model's. On one 80 GB GPU the 8B weights (~16 GB) plus a full-length KV cache (~17 GB) fit.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

pip install --quiet "vllm>=0.6"
exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --generation-config vllm \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json
