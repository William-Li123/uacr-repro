# Reproduction Protocol

Start with the root README for installation and public downloads. All commands
below run from the repository root. `work/` is the default generated-data root;
set `UACR_WORKDIR` before every command to change it. No existing prediction cache
or unpublished trained adapter is required.

The supported command-line entry points are the scripts at the repository root.
Files inside `script/`, `eval/` and `route_compare_test/scripts/` are internal
evaluation/training helpers; call them through the root entry points so paths and
configuration are initialized consistently.

## Protocols

| Protocol | Local detector | Multimodal verifier | SAEC masks |
|---|---|---|---|
| `retrain-local` (default) | Trained from normal training queries | Train from public base weights | Recompute from images and public YOLO weights |
| `fixed-local` | Bundled fixed scores | Train from public base weights | Bundled fixed masks |

The second protocol isolates changes to multimodal training/routing. It is not
an independent reproduction of detector training. Never mix its results with the
first protocol without labeling them. Full retraining is not guaranteed to match
historical numbers bit for bit: GPU kernels and model revisions can differ, and
this implementation monitors EfficientAD on validation images only. It uses the
last training checkpoint, not a checkpoint selected on held-out test results.

## 1. Prepare fixed data

Follow [DATASETS.md](DATASETS.md), then:

```bash
python bootstrap.py --raw-root work/raw --check-only
python bootstrap.py --raw-root work/raw
python reproduce.py check
```

Preparation resolves relative paths, checks duplicate IDs and query overlap,
and builds detector input directories. Those directories call the validation
branch `test` because the upstream trainer expects that name; they contain
project validation queries, not held-out test queries. Reference images remain
the exact images listed in each manifest (typically three, two for KSDD2).
Text targets and original prompts are included in the repository.

The supervised split differs from official unsupervised benchmark protocols.
Do not report these results as standard unsupervised benchmark scores.

## 2. Small smoke run

After downloading the 7B base model, test the data/model interface before a long
run. These outputs are deliberately separate from full evaluation outputs:

```bash
python reproduce.py train --model qwen25vl_7b --dataset mvtec_ad_80p --run smoke --limit 2
python reproduce.py infer --model qwen25vl_7b --dataset mvtec_ad_80p --split val --run smoke --limit 2 --adapter-root work/runs/smoke/smoke_adapters
python reproduce.py infer --model qwen25vl_7b --dataset mvtec_ad_80p --split val --run smoke --limit 2 --base
```

Use the equivalent path under your custom work directory if configured. A tiny
training run tests execution only; it cannot establish convergence or accuracy.

## 3. Complete run

```bash
python pipeline.py --run experiment1 --dry-run
CUDA_VISIBLE_DEVICES=0 bash run_all.sh --run experiment1
```

The pipeline runs each dataset/backbone sequentially on one GPU. For a detached
Linux run, use `nohup bash run_all.sh --run experiment1 > experiment1.log 2>&1 &`.
Monitor the log and GPU memory. Download/model loading time is not inference time.

Stages can also be executed separately:

```bash
python local_detector.py train --dataset mvtec_ad_80p --run experiment1
python local_detector.py score --dataset mvtec_ad_80p --run experiment1
python local_detector.py saec-routes --dataset mvtec_ad_80p --run experiment1
python reproduce.py train --model qwen25vl_7b --dataset mvtec_ad_80p --run experiment1
python reproduce.py infer --model qwen25vl_7b --dataset mvtec_ad_80p --split val --run experiment1
python reproduce.py infer --model qwen25vl_7b --dataset mvtec_ad_80p --split test --run experiment1
```

Repeat inference with `--base` for the unadapted baseline. The pipeline handles
both backbones and all selected datasets automatically. After all predictions:

```bash
python reproduce.py routes --run experiment1 --eff-scores work/runs/experiment1/merged_val_test_scores.csv --saec-run work/runs/experiment1
```

For a subset, pass the same `--datasets` list to preparation, the pipeline and
manual routing. KSDD2 must be accompanied by MVTec AD because it uses that adapter.
KSDD2 adapter selection is fixed in advance, not chosen by test-set accuracy.

## Parameters

- Backbones: Qwen2.5-VL-7B-Instruct and Qwen3.5-9B; see pinned resource revisions.
- Train/inference image budget: 147456 pixels. The 65536 option is an explicitly
  separate evaluation control, not the default aligned training protocol.
- LoRA: rank 8, alpha 16, dropout 0.05; one epoch; batch size 2; gradient
  accumulation 4; learning rate 0.0002; maximum sequence length 4096; seed 42.
- Three-field loss weights: label 10, defect type 3, explanation 1. Normal to
  abnormal sampling ratio: 2, except VisA uses 3.
- Generation: batch size 4, maximum 180 new tokens, the manifest's row prompt.
- EfficientAD: see `configs/efficientad.yaml`; threshold maximizes validation
  balanced accuracy, using accuracy for ties. No test labels calibrate it.
- UACR: validation search over q; objective accuracy + 0.5 TypeHit-A - 0.1
  invocation rate; target invocation cap 0.7. The original implementation's
  fallback when no candidate meets the cap is retained; inspect selected-q output
  rather than assuming every dataset satisfies the cap.
- SAEC: complexity/confidence routing followed by the same trained three-field
  adapter as UACR. Its complexity quantile is computed on evaluated images without
  labels. This is an adapted comparator, not a claim to reproduce all original
  SAEC paper implementation choices.

## Resume and input checks

Completed adapter/local training stages have `TRAINING_DONE` markers. Adapter
training resumes from the last saved checkpoint after interruption. An interrupted
local-detector training directory is rejected; use a new run name, rather than
silently treating partial weights as final. Predictions use the evaluator's cache
and routing verifies exact IDs for both validation and test before summarizing.
Changing pixels, adapter paths or prepared datasets requires a new output run or
work directory. Do not modify model contents in place and reuse old predictions.

## Results and limitations

`work/runs/<run>/metrics.csv` combines Accuracy, TypeHit-A, TypeHit-P, TypeF1,
ExplainHit and `large_model_invocation_rate`. Internal compatibility columns may
retain older names. Report invocation rates as large-model invocation rates.
Semantic metrics without annotations are empty, not zero. Read
`eval/EVALUATION_RULES.md` for metric denominators and normal-exit behavior.

ExplainHit is a keyword-match heuristic, not a correctness or faithfulness audit.
TypeF1 is the harmonic mean of this project's A/P scores, not standard multiclass
F1. Generation batch throughput is not end-to-end single-image latency. Measure
hardware-specific latency separately with warmup and synchronization.

The full pipeline is supplied, but a fresh public-download-to-full-training run
has not been executed as part of this packaging update. Consult
[VALIDATION.md](../VALIDATION.md) for the checks actually performed.
