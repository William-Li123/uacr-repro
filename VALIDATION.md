# Validation Scope

This document distinguishes packaging checks from a new end-to-end experiment.
Successful code checks do not establish fresh-training accuracy.

## Public-resource revision

- CPU tests pass on Windows/Python 3.12 and Linux/Python 3.10: portable manifests,
  split separation, exact fixed-table coverage, preparation/re-run behavior,
  exclusion of held-out queries from detector training/monitoring, CLI help and
  both pipeline command plans.
- All Python source files compile.
- The new public entry point was used to replay existing predictions with the
  converted manifests: UACR's 12 summary rows and SAEC's 10 rows matched the
  reference counts, accuracy, TypeHit-A/P, ExplainHit and invocation counts/rates
  within 1e-12. This confirms evaluation wiring, not fresh generation quality.
- All 29,041 manifest rows across 15 splits resolve to existing original dataset
  images in the development validation environment. Query/reference choices,
  IDs, prompts and targets are retained. This checks the original-layout mapping;
  it is not a fresh download of every dataset archive.
- Public EfficientAD teacher and YOLO classification files were actually
  downloaded. Their SHA-256 values are enforced by the resource downloader.
- Qwen public repository revisions are pinned. The pins identify this repository
  version; they are not asserted to identify historical model downloads.
- Current source, manifests and documentation are checked for developer-specific
  storage paths. Generated local runtime paths are not committed.

## Historical checks

Files in `provenance/` record earlier checks, not completion of fresh training for
this revision. Earlier cached-result replay matched UACR (12 rows including two
weighted totals) and SAEC (10 rows) at tolerance 1e-12 for the recorded fields.
An earlier 7B two-image GPU inference smoke test completed. An earlier EfficientAD
forward pass produced a finite score and the same binary decision as its reference,
but its absolute score difference was 0.00019151, exceeding 1e-4. These are limited
execution checks, not evidence of numerical identity across hardware.

## Not certified by this update

- A clean virtual-environment dependency installation from scratch.
- Fresh downloading/extracting of every complete dataset and both base models.
- All local-detector training runs and six/eight adapter training runs.
- A complete fresh 7B/9B validation/test generation run after that training.
- Exact agreement of freshly trained models with earlier report values.
- GPU training on macOS or lower-memory GPUs.

Use the small smoke commands first, retain environment/configuration records, and
label `retrain-local` versus `fixed-local` results explicitly.
