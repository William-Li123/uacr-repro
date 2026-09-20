from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__import__("os").environ["UACR_ROOT"])
SCRIPT_ROOT = ROOT / "route_compare_test" / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import tune_per_dataset_q_20260630 as route  # noqa: E402
from eval_three_field_qwen import aggregate, norm_type, parse_prediction, row_metrics  # noqa: E402


MODELS = ["qwen35_9b", "qwen25vl_7b"]
DATASETS = ["goodsad_80p", "mvtec_ad_80p", "mvtec_loco_80p", "visa_80p", "ksdd2_mvtlike"]
Q_GRID = [round(float(x), 2) for x in np.arange(0.03, 1.0001, 0.01)]
QWEN_RATE_CAP = 0.70
ACC_WEIGHT = 1.0
TYPE_WEIGHT = 0.5
RATE_WEIGHT = 0.1


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def predictions(cache_root: Path, model: str, dataset: str, split: str) -> dict[str, dict[str, Any]]:
    path = cache_root / model / dataset / split / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return {str(row["id"]): row for row in read_csv(path)}


def add_type_p(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not any(int(float(row.get("has_type_target") or 0)) for row in rows):
        summary["defect_type_hit_pred_abnormal"] = None
        summary["denom_type_pred_abnormal"] = 0
        return summary

    pred_abnormal_rows = [
        row for row in rows
        if str(row.get("pred_label") or "").strip().lower() == "abnormal"
    ]
    hits = sum(int(float(row.get("defect_type_hit") or 0)) for row in pred_abnormal_rows)
    pred_abnormal = len(pred_abnormal_rows)
    summary["defect_type_hit_pred_abnormal"] = hits / pred_abnormal if pred_abnormal else None
    summary["denom_type_pred_abnormal"] = pred_abnormal
    return summary


def weighted_total(rows: list[dict[str, Any]], model: str, variant: str) -> dict[str, Any]:
    n = sum(int(row.get("n", 0) or 0) for row in rows)
    normal_den = sum(int(row.get("normal_n", 0) or 0) for row in rows)
    abnormal_den = sum(int(row.get("abnormal_n", 0) or 0) for row in rows)
    type_a_den = sum(int(row.get("denom_type_abnormal", 0) or 0) for row in rows)
    type_p_den = sum(int(row.get("denom_type_pred_abnormal", 0) or 0) for row in rows)
    explain_den = sum(int(row.get("denom_explanation_abnormal", 0) or 0) for row in rows)
    qwen_called = sum(int(row.get("qwen_called", 0) or 0) for row in rows)

    def weighted(key: str, den_key: str = "n") -> float:
        den = sum(float(row.get(den_key, 0) or 0) for row in rows)
        if not den:
            return 0.0
        return sum(float(row.get(key, 0) or 0) * float(row.get(den_key, 0) or 0) for row in rows) / den

    return {
        "variant": variant,
        "model_key": model,
        "dataset": "weighted_total",
        "n": n,
        "label_accuracy": weighted("label_accuracy"),
        "balanced_accuracy": weighted("balanced_accuracy"),
        "normal_recall": weighted("normal_recall", "normal_n") if normal_den else 0.0,
        "abnormal_recall": weighted("abnormal_recall", "abnormal_n") if abnormal_den else 0.0,
        "defect_type_exact_abnormal": weighted("defect_type_exact_abnormal", "denom_type_abnormal") if type_a_den else None,
        "defect_type_hit_pred_abnormal": weighted("defect_type_hit_pred_abnormal", "denom_type_pred_abnormal") if type_p_den else None,
        "explanation_keyword_hit_abnormal": weighted("explanation_keyword_hit_abnormal", "denom_explanation_abnormal") if explain_den else None,
        "qwen_called": qwen_called,
        "qwen_rate": qwen_called / n if n else 0.0,
        "denom_type_abnormal": type_a_den,
        "denom_type_pred_abnormal": type_p_den,
        "denom_explanation_abnormal": explain_den,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--variant-name", required=True)
    args = parser.parse_args()
    args.result_root.mkdir(parents=True, exist_ok=True)

    eff_scores = route.load_eff_scores()
    all_candidates: list[dict[str, Any]] = []
    all_selected: list[dict[str, Any]] = []
    all_test: list[dict[str, Any]] = []

    for model in MODELS:
        model_rows: list[dict[str, Any]] = []
        for dataset in DATASETS:
            val_manifest = route.read_jsonl(route.DATA_ROOT / dataset / "val.jsonl")
            val_rows = []
            for row in val_manifest:
                item = dict(row)
                item["dataset_key"] = dataset
                val_rows.append(item)
            val_preds = predictions(args.cache_root, model, dataset, "val")

            candidates = []
            for q in Q_GRID:
                radii = route.radius_by_category(val_rows, eff_scores, q)
                routed = route.evaluate_split(
                    val_rows, "val", dataset, model, "ours", val_preds, eff_scores,
                    radii, "ours_pixel_aligned_val", parse_prediction, norm_type, row_metrics,
                )
                summary = route.summarize(model, "ours_pixel_aligned_val", dataset, routed, aggregate)
                summary["score"] = (
                    ACC_WEIGHT * float(summary.get("label_accuracy", 0.0))
                    + TYPE_WEIGHT * float(summary.get("defect_type_exact_abnormal", 0.0))
                    - RATE_WEIGHT * float(summary.get("qwen_rate", 0.0))
                )
                summary["q"] = q
                summary["eligible"] = int(float(summary["qwen_rate"]) <= QWEN_RATE_CAP)
                all_candidates.append({"variant": args.variant_name, **summary})
                candidates.append(summary)

            eligible = [row for row in candidates if row["eligible"]]
            pool = eligible or candidates
            selected = sorted(
                pool,
                key=lambda row: (
                    float(row["score"]),
                    float(row.get("label_accuracy", 0.0)),
                    float(row.get("defect_type_exact_abnormal", 0.0)),
                    -float(row.get("qwen_rate", 0.0)),
                ),
                reverse=True,
            )[0]
            all_selected.append({"variant": args.variant_name, **selected})

            radii = route.radius_by_category(val_rows, eff_scores, float(selected["q"]))
            test_rows = route.read_jsonl(route.DATA_ROOT / dataset / "test.jsonl")
            test_preds = predictions(args.cache_root, model, dataset, "test")
            routed_test = route.evaluate_split(
                test_rows, "test", dataset, model, "ours", test_preds, eff_scores,
                radii, "ours_pixel_aligned", parse_prediction, norm_type, row_metrics,
            )
            write_csv(args.result_root / model / f"{dataset}.predictions.csv", routed_test)
            test_summary = route.summarize(model, "ours_pixel_aligned", dataset, routed_test, aggregate)
            test_summary = add_type_p(test_summary, routed_test)
            test_summary.update({"variant": args.variant_name, "selected_q": selected["q"]})
            all_test.append(test_summary)
            model_rows.append(test_summary)

        all_test.append(weighted_total(model_rows, model, args.variant_name))

    write_csv(args.result_root / "val_candidates.csv", all_candidates)
    write_csv(args.result_root / "selected_q.csv", all_selected)
    write_csv(args.result_root / "test_summary.csv", all_test)
    config = {
        "variant": args.variant_name,
        "objective": "1.0 * Accuracy + 0.5 * TypeHit-A - 0.1 * LargeModelRate",
        "q_grid": "0.03..1.00 step 0.01",
        "large_model_rate_cap": QWEN_RATE_CAP,
        "selection_split": "validation",
        "evaluation_split": "test",
        "cache_root": str(args.cache_root),
    }
    (args.result_root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({"result_root": str(args.result_root), "summary": str(args.result_root / "test_summary.csv")}, indent=2))


if __name__ == "__main__":
    main()
