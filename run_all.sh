#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"
PY="${PY:-/data/yuzheng/miniconda3/envs/detection/bin/python}"
RUN="${RUN:-full_reproduction}"
case "$RUN" in ''|.|..|*/*) echo 'RUN must be a directory name' >&2; exit 2 ;; esac
mkdir -p "$ROOT/runs/$RUN"
exec > >(tee -a "$ROOT/runs/$RUN/pipeline.log") 2>&1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
"$PY" reproduce.py check
for model in qwen25vl_7b qwen35_9b; do
  for dataset in goodsad_80p mvtec_ad_80p mvtec_loco_80p visa_80p; do
    if [[ ! -f "$ROOT/runs/$RUN/adapters/$model/$dataset/TRAINING_DONE" ]]; then
      "$PY" reproduce.py train --model "$model" --dataset "$dataset" --run "$RUN"
    fi
  done
  for dataset in goodsad_80p mvtec_ad_80p mvtec_loco_80p visa_80p ksdd2_mvtlike; do
    for split in val test; do
      "$PY" reproduce.py infer --model "$model" --dataset "$dataset" --split "$split" --run "$RUN" --adapter-root "$ROOT/runs/$RUN/adapters"
      "$PY" reproduce.py infer --model "$model" --dataset "$dataset" --split "$split" --run "$RUN" --base
    done
  done
done
"$PY" reproduce.py routes --run "$RUN"
echo 'DONE: adapters retrained, val/test predictions generated, UACR/SAEC recomputed with frozen archived local detector.'
