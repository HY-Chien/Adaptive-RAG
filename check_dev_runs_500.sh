#!/usr/bin/env bash
set -euo pipefail

# 這個是你在 runner 裡用的 MODEL 名稱（命令列參數）
MODEL="flan-t5-xl"

# 把 MODEL 轉成資料夾裡用的格式：flan_t5_xl
MODEL_DIR="${MODEL//-/_}"

SYSTEMS=("ircot_qa" "oner_qa" "nor_qa")
DATASETS=("nq" "squad" "trivia" "hotpotqa" "2wikimultihopqa" "musique")

echo "Checking DEV (dev_500) runs for MODEL=${MODEL} (dir name = ${MODEL_DIR})"
echo

for DATASET in "${DATASETS[@]}"; do
  for SYSTEM in "${SYSTEMS[@]}"; do

    # instantiated_configs / predictions 裡實驗名稱的前綴
    EXP_PREFIX="${SYSTEM}_${MODEL_DIR}_${DATASET}____prompt_set_1"

    # 在 predictions/dev_500 底下找 evaluation_metrics__*dev_500_subsampled.json
    METRICS_FILE=$(find predictions/dev_500 -type f \
      -path "*${EXP_PREFIX}*/evaluation_metrics__*dev_500_subsampled.json" \
      2>/dev/null | head -n 1 || true)

    if [[ -n "${METRICS_FILE}" ]]; then
      echo "[OK ] SYSTEM=${SYSTEM}, DATASET=${DATASET}"
      echo "     metrics: ${METRICS_FILE}"
    else
      # 看看 experiment dir 在不在
      EXP_DIR=$(find predictions/dev_500 -type d -path "*${EXP_PREFIX}*" \
        2>/dev/null | head -n 1 || true)

      echo "[MISS] SYSTEM=${SYSTEM}, DATASET=${DATASET}"
      if [[ -n "${EXP_DIR}" ]]; then
        echo "     (experiment dir exists but NO dev_500 metrics)"
        echo "     dir: ${EXP_DIR}"
      else
        echo "     (no experiment dir for prefix: ${EXP_PREFIX})"
      fi
    fi

    echo
  done
done
