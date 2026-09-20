# Three-field Qwen Evaluation Rules

This document defines the evaluation protocol for the three-field anomaly-inspection Qwen data under:

```text
/data/yuzheng/anomaly_detection2/data/three_field_qwen
```

The goal is to evaluate whether a Qwen base model or Qwen LoRA adapter can:

1. classify an image as `normal` or `abnormal`;
2. output the correct dataset/category-specific `defect_type`;
3. provide a concise and visually grounded `explanation`.

The protocol is intentionally narrower than the older strong-JSON protocol. The model is no longer asked to generate `dataset`, `category`, `anomaly_family`, `location`, or `visual_evidence` as separate fields. Those concepts may still be reflected inside the free-text `explanation`, but they are not standalone output fields.

## Data Splits

Each dataset directory contains exactly three files:

```text
<dataset>/
  train.jsonl
  val.jsonl
  test.jsonl
```

The splits are inherited from the previous experiments:

- `train`: the same Qwen SFT training split used before.
- `val`: the fixed validation manifest from `typehit_optimization_v5_full_short_loco`.
- `test`: the fixed holdout test manifest from `typehit_optimization_v5_full_short_loco`.

Do not re-split these files during evaluation. Use `val` for prompt/loss/checkpoint/hyperparameter selection and `test` only for final reporting.

## Required Output Format

Every prediction must be one compact JSON object with exactly three keys:

```json
{
  "label": "normal",
  "defect_type": "good",
  "explanation": "The query matches the normal references and shows no visible defect."
}
```

Required key semantics:

| Key | Allowed values / meaning |
|---|---|
| `label` | Exactly `normal` or `abnormal`. |
| `defect_type` | Must be one of the sample's `allowed_defect_types`. For normal samples, it must be `good`. |
| `explanation` | One concise sentence describing the visible evidence. For normal samples, it should explain consistency with normal references. |

KSDD2 exception:

```json
{
  "label": "abnormal",
  "defect_type": "",
  "explanation": ""
}
```

KSDD2 has no reliable fine-grained defect type or textual explanation in this package, so only `label` is meaningful. `defect_type` and `explanation` are intentionally empty strings for all KSDD2 samples.

## Prompt Design

The evaluation script defaults to:

```text
--prompt-source row
```

This means the script uses the `prompt` field already stored in each JSONL sample. This is the preferred mode because it keeps training and evaluation aligned. The same prompt text used to construct supervised training samples is reused for evaluation.

The prompt has the following structure:

```text
You are a careful industrial visual anomaly inspector.
Dataset: <dataset name>. Category: <category>.
The first <N> image(s) are normal reference examples from the same category; the last image is the query image.
Compare the query with the references and decide whether it is normal or abnormal.
Allowed defect_type values are: <allowed list>.
Return only one compact JSON object with exactly three keys: "label", "defect_type", and "explanation".
The label must be exactly "normal" or "abnormal".
For normal images, defect_type must be "good".
For abnormal images, choose the best allowed defect_type and write one concise sentence explaining the visible evidence.
```

MVTec LOCO adds a coarse-label rule:

```text
MVTec LOCO uses coarse defect_type labels:
logical_anomaly means wrong count, missing or extra object, wrong object, or wrong arrangement;
structural_anomaly means local damage, contamination, deformation, or surface/appearance change.
```

KSDD2 adds an empty-field rule:

```text
For KSDD2 here, "defect_type" and "explanation" must be empty strings because no reliable fine-grained textual labels are available.
```

Prompt design principles:

1. Normal references appear before the query image.
2. The last image is always the query image.
3. The allowed `defect_type` list is explicitly provided.
4. The model is not allowed to invent extra JSON fields.
5. The normal-case output is constrained to `defect_type="good"`.
6. The abnormal-case output must include one concise visual explanation.

## Parsing Rules

The evaluator extracts the first valid JSON object from the generated text. Markdown fences are stripped if present.

The evaluator records:

| Metric field | Meaning |
|---|---|
| `valid_json` | A JSON object was parsed from the output. |
| `exact_schema` | The parsed JSON has exactly `label`, `defect_type`, and `explanation`, with no extra keys. |
| `pred_label` | Parsed and normalized label. Unknown labels are marked `unknown`. |
| `pred_defect_type` | Lowercased, underscore-normalized defect type. |
| `pred_explanation` | Raw explanation string from JSON. |

If JSON parsing fails, the evaluator still tries a conservative fallback for `label` by searching for `normal`, `abnormal`, `defect`, or `anomaly` in the raw text, but `valid_json=0` and `exact_schema=0`.

## Metrics

### Format Metrics

```text
valid_json_rate = valid JSON outputs / all samples
exact_schema_rate = outputs with exactly three required keys / all samples
allowed_type_rate = outputs whose defect_type obeys the allowed list / all samples
```

For KSDD2, `allowed_type_rate` requires `defect_type=""`.

### Label Metrics

```text
label_accuracy = correct normal/abnormal labels / all samples
normal_recall = normal samples predicted normal / all normal samples
abnormal_recall = abnormal samples predicted abnormal / all abnormal samples
balanced_accuracy = (normal_recall + abnormal_recall) / 2
```

Balanced accuracy is important because most datasets are class-imbalanced.

### Defect Type Metrics

These are computed only on true abnormal samples with a non-empty target `defect_type`.

```text
defect_type_exact_abnormal = exact defect_type match / eligible abnormal samples
defect_type_hit_abnormal = overlap match / eligible abnormal samples
```

Exact match requires the full normalized string to match.

Hit match allows overlap when a target has multiple defect types joined by `__`. For example:

```text
target: chunk_of_wax_missing__damaged_corner_of_packaging
prediction: damaged_corner_of_packaging
```

This counts as a hit but not an exact match.

### Explanation Metric

The current automatic explanation metric is:

```text
explanation_keyword_hit_abnormal
```

It is computed only on true abnormal samples with a non-empty target explanation.

Procedure:

1. Extract informative keywords from the target `defect_type` and target `explanation`.
2. Extract informative keywords from the predicted `explanation`.
3. Count a hit if the overlap is at least one keyword for short targets, or at least two keywords for longer targets.

This is a lightweight lexical metric. It is not a full semantic judge. It should be reported as a proxy for explanation grounding, not as a definitive natural-language quality score.

Recommended final explanation reporting:

```text
valid_json_rate
exact_schema_rate
defect_type_hit_abnormal
explanation_keyword_hit_abnormal
```

For human-facing examples, also inspect raw predictions from `predictions.csv`.

## Output Files

The evaluator writes:

```text
<out-dir>/
  run_config.json
  predictions.csv
  summary.json
  summary.by_category.csv
  summary.by_label.csv
```

`predictions.csv` contains one row per sample, including the raw model answer, parsed JSON fields, target fields, and all per-sample metrics.

`summary.json` contains:

- `overall`
- `by_category`
- `by_label`

## Example Commands

Evaluate Qwen base on MVTec AD test:

```bash
python /data/yuzheng/anomaly_detection2/eval/eval_three_field_qwen.py \
  --dataset mvtec_ad_80p \
  --split test \
  --model /data/yuzheng/anomaly_detection2/model_qwen_base/Qwen3.5-9B \
  --out-dir /data/yuzheng/anomaly_detection2/eval/outputs/base_mvtec_ad_test \
  --batch-size 4
```

Evaluate a LoRA adapter:

```bash
python /data/yuzheng/anomaly_detection2/eval/eval_three_field_qwen.py \
  --dataset goodsad_80p \
  --split test \
  --model /data/yuzheng/anomaly_detection2/model_qwen_base/Qwen3.5-9B \
  --adapter /path/to/lora_adapter \
  --out-dir /data/yuzheng/anomaly_detection2/eval/outputs/lora_goodsad_test \
  --batch-size 4
```

Run a small smoke test:

```bash
python /data/yuzheng/anomaly_detection2/eval/eval_three_field_qwen.py \
  --dataset mvtec_loco_80p \
  --split val \
  --out-dir /data/yuzheng/anomaly_detection2/eval/outputs/smoke_loco_val \
  --limit 20
```

## Reporting Rules

When reporting results, use the same split and prompt source for all compared models:

```text
prompt-source = row
split = test for final results
greedy decoding: do_sample=False
```

Do not compare a base model evaluated with one prompt against a LoRA model evaluated with another prompt. If prompt changes are tested, report them as a separate ablation.
