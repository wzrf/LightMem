#!/bin/bash

CUDA_VISIBLE_DEVICES=1 \
FUSIONRAG=false \
PYTHONPATH=../../src/ \
python3 search_locomo.py \
  --total-limit 60 \
  --retrieval-mode combined \
  --embedder huggingface \
  --embedding-model-path /mnt/qjhs-sh-lab-01/models/all-MiniLM-L6-v2/ \
  --llm-api-key sk-11ce7640e46049a6977c0d96ba855ffb \
  --llm-base-url http://127.0.0.1:30004/v1 \
  --llm-model kimi-k2.6 \
  --judge-api-key sk-11ce7640e46049a6977c0d96ba855ffb \
  --judge-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --judge-model deepseek-v3.2 \
  --enable-summary \
  --qdrant-dir ./qdrant_post_update_Kimi-K2.6/ \
  --output-dir ../lightmem_locomo_results_Kimi-K2.6/