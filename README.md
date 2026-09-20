# UACR reproducible core

## Start here

Chinese instructions: [开始复现.md](开始复现.md). Verification scope:
[VALIDATION.md](VALIDATION.md).

This repository contains code, fixed experiment manifests, archived local scores,
SAEC routing masks and verification records. It does **not** contain image datasets,
model weights, full prediction caches, generated runs or credentials. The verification
records describe tests already performed on the source server, not a fresh-clone test.

On the original server, `/data/yuzheng/uacr_repro` is already prepared. On another
server with both archives mounted, restore a separate runtime directory:

```bash
git clone https://github.com/William-Li123/uacr-repro.git uacr-repro-source
cd uacr-repro-source
python bootstrap.py --archive-parent /mnt/aoss-250010161/cloudplatform/models \
  --dest /data/yuzheng/uacr_repro
```

The destination must not already exist. Bootstrap copies code and prepares manifests
and resource links; it does not download the archives or install dependencies.
Use the environment described below, then run `reproduce.py check` and `replay`.
`verify_replay.py` requires the `archived147` replay output to exist first.

This project consolidates the final three-field, pixel-aligned pipeline from the two September 3 archives. Source archives are never modified. Python implementation files are local copies, not imports from the archives. `provenance/source_files.json` records original and extracted checksums.

## Environment and resources

```bash
cd /data/yuzheng/uacr_repro
PY=/data/yuzheng/miniconda3/envs/detection/bin/python
$PY reproduce.py check
```

`resources/` contains explicit links to immutable archived models and prediction caches. Prepared JSONL manifests are local and retain the original IDs, splits, prompts, targets, and reference choices; only image paths are relocated. Original images remain on the mounted archive. This is a clean CODE distribution, not a second copy of 100+ GB of model/data assets. Copying only this directory to another server requires remapping the resource links and image paths, or running `bootstrap.py` against the same archives. Keep the mount available during training/inference.

## 1. Reproduce results without new inference (CPU)

```bash
$PY reproduce.py replay --run archived147
# Optional older low-resolution ablation:
$PY reproduce.py replay --pixels 65536 --run archived65
```

Outputs: `runs/<run>/uacr/{selected_q,val_candidates,test_summary}.csv`, `runs/<run>/saec/test_summary.csv`, and `runs/<run>/metrics.csv`. TypeF1 is computed from TypeHit-A/P; missing semantic targets remain missing, not fabricated scores. Rates are fractions, not percentages. Per-image output is also retained.

## 2. Smoke test and full inference with final archived adapters

```bash
CUDA_VISIBLE_DEVICES=0 $PY reproduce.py infer --model qwen25vl_7b --dataset mvtec_ad_80p --limit 2 --run smoke
for model in qwen25vl_7b qwen35_9b; do
  for dataset in goodsad_80p mvtec_ad_80p mvtec_loco_80p visa_80p ksdd2_mvtlike; do
    for split in val test; do
      CUDA_VISIBLE_DEVICES=0 $PY reproduce.py infer --model "$model" --dataset "$dataset" --split "$split" --run fresh147
    done
  done
done
$PY reproduce.py routes --run fresh147
```

Base-model comparison: add `--base` to the same inference commands. Same three-field row prompt, same three references, same pixel budget. Base predictions go to a separate directory and their `summary.json` contains the metrics. `--limit` always writes into isolated smoke directories and cannot pass the full-cache check.

## 3. Train eight aligned three-field adapters, then evaluate

```bash
for model in qwen25vl_7b qwen35_9b; do
  for dataset in goodsad_80p mvtec_ad_80p mvtec_loco_80p visa_80p; do
    CUDA_VISIBLE_DEVICES=0 $PY reproduce.py train --model "$model" --dataset "$dataset" --run retrain147
  done
done
# Repeat the inference loop above with:
# --run retrain147 --adapter-root /data/yuzheng/uacr_repro/runs/retrain147/adapters
$PY reproduce.py routes --run retrain147
```

Training: 147456 maximum pixels/image, one epoch, batch 2, gradient accumulation 4, learning rate 2e-4, LoRA rank 8 / alpha 16 / dropout 0.05, seed 42, field loss weights 10:3:1. Normal:abnormal sampling is 2:1 except VisA 3:1. Inference uses the same 147456 budget, batch 4, max 180 new tokens. Checkpoint resume is supported without deleting incomplete runs. GPU nondeterminism can prevent bit-identical retraining.

One H100 is sufficient for sequential execution. For two GPUs, run one model's loop per GPU; do not share output files between workers. The archived two-GPU scheduler and destructive cleanup scripts are intentionally not carried over.

## 4. EfficientAD training, scoring and SAEC routing from images

The paper-result replay above intentionally uses the archived detector. To build a NEW full experiment, use separate outputs:

```bash
for dataset in goodsad_80p mvtec_ad_80p mvtec_loco_80p visa_80p ksdd2_mvtlike; do
  CUDA_VISIBLE_DEVICES=0 $PY local_detector.py train --dataset "$dataset" --run local_retrain
  CUDA_VISIBLE_DEVICES=0 $PY local_detector.py score --dataset "$dataset" --run local_retrain --weights-root /data/yuzheng/uacr_repro/runs/local_retrain/efficientad
  $PY local_detector.py saec-routes --dataset "$dataset" --run local_retrain
done
$PY reproduce.py routes --run retrain147_local \
  --cache-root /data/yuzheng/uacr_repro/runs/retrain147/predictions \
  --eff-scores /data/yuzheng/uacr_repro/runs/local_retrain/merged_val_test_scores.csv \
  --saec-run /data/yuzheng/uacr_repro/runs/local_retrain
```

Use this NEW run name: `retrain147` already records the archived local scores in
`inputs.json`, so changing those inputs in the same run is intentionally rejected.
The new comparison reuses the newly inferred adapter predictions. Base-model
summaries remain under `runs/retrain147/base_predictions/`; they are not
automatically copied into the new run's combined table.

Use `local_detector.py score` WITHOUT `--weights-root` to re-score using the archived trained detector. Thresholds are refitted on validation labels only. Local training uses the archived category data layout, pretrained teacher, Imagenette penalty set, 5001 iterations, 256px input, seed 42. The legacy trainer internally splits normal training images 80/20 for fitting and quantile calibration. Its evaluation monitoring uses the dataset test directory; the new wrapper deliberately chooses the final (`*_last`) weights, NOT the highest test-AUROC checkpoint. Therefore this retraining path is a new run, not a guarantee of bit-identical historical weights. The underlying trainer's test monitoring is retained and disclosed, not described as validation-only training.

SAEC rebuild uses archived YOLO classification weights and original complexity/confidence code. It does not use EfficientAD as its local branch. All generated masks are isolated in the new run. If the image data cannot be decoded, fix the input instead of accepting a zero complexity score.

## Environment restoration

`requirements.lock.txt` records versions in the current detection environment, not a claim that the original training environment was independently reconstructed. Use Python 3.10 and install the matching CUDA-enabled PyTorch build first, then these pinned packages. Development-version packages may require their matching upstream source distribution rather than a public PyPI wheel. The existing environment is the verified execution path; do not upgrade it in place.

```bash
$PY environment.py
$PY verify_replay.py
```

## Calibration, fairness and scope

- UACR chooses q on validation only: accuracy + 0.5 TypeHit-A - 0.1 invocation rate; grid 0.03 to 1.00 in 0.01 increments, rate cap 0.70. The preserved implementation falls back to all candidates if none is eligible; inspect `eligible` in selected_q.csv.
- SAEC here means the archived SAEC routing variant with the SAME aligned three-field adapter as UACR, not the original binary-adapter SAEC. Its cached routing masks and complexity implementation are preserved. See `route_compare_test/scripts/prepare_cached_baseline_routes.py` for the evaluation-split complexity quantile behavior; do not describe this as validation-calibrated SAEC.
- KSDD2 has a normal-image training manifest but no dedicated three-field adapter in this experiment. It uses the fixed MVTec AD transfer adapter for both routes. Historical test-selected adapter sweeps are deliberately not the canonical path. Do not choose an adapter using test accuracy and report it as unbiased held-out validation.
- Full val/test inference does not include training samples. The original split is retained; no new split is generated.
- `--datasets mvtec_ad_80p mvtec_loco_80p visa_80p ksdd2_mvtlike` limits routing/summary to the four paper datasets. Archives also retain GoodsAD for completeness.
- Cached prediction timings are not measured end-to-end serving latency. This package does not fabricate new latency values.
- `legacy/tools/EfficientAD` preserves the actual local-detector implementation; `legacy/configs` and its analyzer are included. Canonical route replay uses the archived EfficientAD score table and frozen local weights. Retraining the local detector changes scores and requires regenerating thresholds and routing masks before comparison, not reusing the archived table.
- `script/build_three_field_qwen_data.py` is retained for provenance, not called automatically: prepared manifests are the experiment's authoritative data version.

Use `--dry-run` to inspect GPU command lines before execution. No training is started during extraction or CPU replay.
