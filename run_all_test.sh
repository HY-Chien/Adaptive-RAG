#!/usr/bin/env bash
set -euo pipefail

MODEL=flan-t5-xl
LLM_PORT_NUM=8010

SYSTEMS=("ircot_qa" "oner_qa" , "nor_qa")
DATASETS=("nq" "squad" "trivia" "hotpotqa" "2wikimultihopqa" "musique")

for DATASET in "${DATASETS[@]}"; do
  for SYSTEM in "${SYSTEMS[@]}"; do
    EXP_NAME="${SYSTEM}_${MODEL//-/_}_${DATASET}____prompt_set_1"
    EXP_DIR="predictions/test/${EXP_NAME}"

    if [[ -d "${EXP_DIR}" ]]; then
      echo "[SKIP] 已經有資料夾，可能跑過：${EXP_DIR}"
      continue
    fi

    echo "=============================="
    echo "Running TEST (補跑缺的)"
    echo "  SYSTEM  = ${SYSTEM}"
    echo "  MODEL   = ${MODEL}"
    echo "  DATASET = ${DATASET}"
    echo "  PORT    = ${LLM_PORT_NUM}"
    echo "=============================="
    echo

    bash run_retrieval_test.sh "${SYSTEM}" "${MODEL}" "${DATASET}" "${LLM_PORT_NUM}"

    echo
    echo "✅ Finished: SYSTEM=${SYSTEM}, DATASET=${DATASET}"
    echo
  done
done
