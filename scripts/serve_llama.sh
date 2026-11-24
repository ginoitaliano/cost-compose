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

pip install --quiet "vllm>=0.6"
exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json
