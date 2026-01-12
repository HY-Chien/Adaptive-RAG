#!/usr/bin/env bash
set -euo pipefail

# 固定的 LLM model & port
MODEL="flan-t5-xl"
LLM_PORT_NUM=8010   # 要跟你開 llm_server 時用的一樣

# 要跑的系統：multi / single / zero
SYSTEMS=("ircot_qa" "oner_qa" "nor_qa")

# 要跑的 datasets
DATASETS=("nq" "squad" "trivia" "hotpotqa" "2wikimultihopqa" "musique")

for DATASET in "${DATASETS[@]}"; do
  for SYSTEM in "${SYSTEMS[@]}"; do
    echo "====================================="
    echo "Running DEV:"
    echo "  SYSTEM  = ${SYSTEM}"
    echo "  MODEL   = ${MODEL}"
    echo "  DATASET = ${DATASET}"
    echo "  PORT    = ${LLM_PORT_NUM}"
    echo "====================================="
    echo

    bash run_retrieval_dev.sh "${SYSTEM}" "${MODEL}" "${DATASET}" "${LLM_PORT_NUM}"

    echo
    echo "✅ Finished DEV: SYSTEM=${SYSTEM}, DATASET=${DATASET}"
    echo
  done
done
