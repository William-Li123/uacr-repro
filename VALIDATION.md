# Verification on 2026-09-20

## Passed

- All 15 prepared JSONL manifests were read; every query/reference path exists.
- No query-image path overlap was found between train, validation and test within a dataset. This is a path-level check, not a perceptual duplicate-image audit.
- Full archived val/test prediction IDs match the prepared manifests for both model sizes and all five datasets.
- Recomputed UACR (12 summary rows, including totals) and same-adapter SAEC (10 rows) match the archived accuracy, TypeHit-A/P, ExplainHit, invocation counts/rates within absolute tolerance 1e-12. See `provenance/replay_verification.json`.
- Qwen2.5-VL-7B plus aligned MVTec AD adapter completed a real two-image inference test. See `runs/smoke/smoke_predictions/` and `provenance/gpu_smoke.log`.
- EfficientAD executed on CUDA. One-image score: 0.8740012049674988; archived score: 0.8741927146911621; difference: 0.00019150972366333008. The threshold decision is unchanged. This is NOT a bit-identical forward reproduction and does NOT pass a 1e-4 score tolerance. See `provenance/local_smoke.json`.
- Python syntax compilation, shell syntax, training CLI imports and command construction were checked.

## Not Executed During Packaging

- Full adapter retraining, full new val/test inference, 9B GPU smoke, and full EfficientAD retraining.
- Regeneration of all SAEC YOLO masks (the current result verification uses the archived masks).
- End-to-end latency benchmarking or independent dependency installation in a blank environment.

The current machine exposes one H100 80GB GPU. Loading the 7B base directly from the archive mount took approximately ten minutes; this is storage/model initialization time, not per-image inference latency. The package does not launch full training in the background.

The canonical replay intentionally uses the original fixed MVTec AD transfer adapter for KSDD2. Later test-selected KSDD2 adapter-sweep tables are not silently substituted into this validation.
