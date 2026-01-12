#!/usr/bin/env bash
set -euo pipefail

# 這個是你在 runner 裡用的 MODEL 名稱（命令列參數）
MODEL="flan-t5-xl"

# 把 MODEL 轉成資料夾裡用的格式：flan_t5_xl
MODEL_DIR="${MODEL//-/_}"

SYSTEMS=("ircot_qa" "oner_qa" "nor_qa")
DATASETS=("nq" "squad" "trivia" "hotpotqa" "2wikimultihopqa" "musique")

echo "Checking test runs (1/10 data) for MODEL=${MODEL} (dir name = ${MODEL_DIR})"
echo

for DATASET in "${DATASETS[@]}"; do
  for SYSTEM in "${SYSTEMS[@]}"; do

    EXP_NAME="${SYSTEM}_${MODEL_DIR}_${DATASET}____prompt_set_1"
    EXP_DIR="predictions/test/${EXP_NAME}"

    # 找 evaluation_metrics__*__test_subsampled.json
    METRICS_FILE="$(ls "${EXP_DIR}"/evaluation_metrics__*__test_subsampled.json 2>/dev/null || true)"

    if [[ -n "${METRICS_FILE}" ]]; then
      echo "[OK ] SYSTEM=${SYSTEM}, DATASET=${DATASET}"
      echo "     metrics: ${METRICS_FILE}"
    else
      echo "[MISS] SYSTEM=${SYSTEM}, DATASET=${DATASET}"
      if [[ -d "${EXP_DIR}" ]]; then
        echo "     (experiment dir exists: ${EXP_DIR}, but no metrics json)"
      else
        echo "     (no experiment dir: ${EXP_DIR})"
      fi
    fi
    echo
  done
done
