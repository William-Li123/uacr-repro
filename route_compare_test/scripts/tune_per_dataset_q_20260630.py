from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__import__("os").environ["UACR_ROOT"])
DATA_ROOT = ROOT / "data" / "three_field_qwen"
OUT_ROOT = ROOT / "route_compare_test"
RESULT_ROOT = OUT_ROOT / "per_dataset_q_tuning_20260630"
EFF_SCORES = ROOT / "data" / "splits" / "hybrid_unified_five" / "merged_val_test_scores.csv"

DATASETS = ["goodsad_80p", "mvtec_ad_80p", "mvtec_loco_80p", "visa_80p", "ksdd2_mvtlike"]
MODEL_KEYS = ["qwen35_9b", "qwen25vl_7b"]
RANDOM_RATES = [0.40, 0.50, 0.60]
RANDOM_SEED = "per_dataset_q_random_20260630"

REFERENCE_QWEN_RATE = {
    "goodsad_80p": 0.5923,
    "ksdd2_mvtlike": 0.2375,
    "mvtec_ad_80p": 0.6404,
    "mvtec_loco_80p": 0.6737,
    "visa_80p": 0.3165,
}

CSV_DATASET_TO_KEY = {
    "goodsad_80p": "goodsad_80p",
    "mvtec_ad_80p": "mvtec_ad_80p",
    "mvtec_loco_80p": "mvtec_loco_80p",
    "visa_80p": "visa_80p",
    "ksdd2": "ksdd2_mvtlike",
}

WEIGHTS = {
    "acc": 1.00,
    "type_hit": 0.25,
    "qwen_rate_deviation": 0.20,
}
RATE_CAP_EXTRA = 0.08
GLOBAL_RATE_CAP = 0.70


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def stable_uniform(dataset: str, sample_id: str) -> float:
    digest = hashlib.sha256(f"{RANDOM_SEED}:{dataset}:{sample_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16 - 1)


def load_eff_scores() -> dict[tuple[str, str, str], dict[str, Any]]:
    df = pd.read_csv(EFF_SCORES)
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rec in df.to_dict("records"):
        dataset_key = CSV_DATASET_TO_KEY.get(str(rec["dataset"]))
        if dataset_key:
            out[(dataset_key, str(rec["split2"]), str(rec["id"]))] = rec
    return out


def deterministic_normal_prediction(dataset: str) -> dict[str, Any]:
    if dataset == "ksdd2_mvtlike":
        return {"valid_json": 1, "exact_schema": 1, "pred_label": "normal", "pred_defect_type": "", "pred_explanation": ""}
    return {
        "valid_json": 1,
        "exact_schema": 1,
        "pred_label": "normal",
        "pred_defect_type": "good",
        "pred_explanation": "The query matches the normal references and shows no visible defect.",
    }


def deterministic_eff_abnormal_prediction(dataset: str) -> dict[str, Any]:
    return {"valid_json": 1, "exact_schema": 1, "pred_label": "abnormal", "pred_defect_type": "", "pred_explanation": ""}


def load_full_qwen(model_key: str, family: str, dataset: str) -> dict[str, dict[str, Any]]:
    path = OUT_ROOT / "qwen_full_test_outputs" / model_key / family / dataset / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return {str(row["id"]): row for row in read_csv(path)}


def load_val_qwen(model_key: str, dataset: str, eff_scores: dict[tuple[str, str, str], dict[str, Any]]) -> dict[str, dict[str, Any]]:
    path = ROOT / "model_qwen_adapter" / "sft_lora" / model_key / dataset / "val_eval" / "predictions.csv"
    if path.exists():
        return {str(row["id"]): row for row in read_csv(path)}
    if dataset != "ksdd2_mvtlike":
        raise FileNotFoundError(path)
    out: dict[str, dict[str, Any]] = {}
    for (ds, split, sample_id), rec in eff_scores.items():
        if ds == dataset and split == "val":
            pred_int = int(rec.get("qwen_pred_int", 0))
            out[sample_id] = {
                "id": sample_id,
                "pred_label": "abnormal" if pred_int == 1 else "normal",
                "pred_defect_type": "",
                "pred_explanation": "",
                "valid_json": 1,
                "exact_schema": 1,
                "raw_answer": json.dumps({"label": "abnormal" if pred_int == 1 else "normal", "defect_type": "", "explanation": ""}),
                "elapsed_sec": rec.get("qwen_elapsed_sec", 0.0),
            }
    return out


def qwen_prediction_from_csv(row: dict[str, Any], parse_prediction, norm_type) -> dict[str, Any]:
    parsed = parse_prediction(row.get("raw_answer") or "")
    if row.get("pred_label"):
        parsed["pred_label"] = row.get("pred_label")
    if row.get("pred_defect_type") is not None:
        parsed["pred_defect_type"] = norm_type(row.get("pred_defect_type"))
    if row.get("pred_explanation") is not None:
        parsed["pred_explanation"] = row.get("pred_explanation")
    parsed["valid_json"] = int(float(row.get("valid_json", parsed["valid_json"]) or 0))
    parsed["exact_schema"] = int(float(row.get("exact_schema", parsed["exact_schema"]) or 0))
    return parsed


def output_row(
    manifest_row: dict[str, Any],
    pred: dict[str, Any],
    method: str,
    model_key: str,
    dataset: str,
    qwen_called: int,
    route_source: str,
    elapsed_sec: float,
    raw_answer: str,
    row_metrics,
) -> dict[str, Any]:
    metrics = row_metrics(manifest_row, pred)
    return {
        "id": str(manifest_row["id"]),
        "dataset": manifest_row.get("dataset"),
        "dataset_key": dataset,
        "split": "test",
        "category": manifest_row.get("category"),
        "image": manifest_row.get("image"),
        "method": method,
        "model_key": model_key,
        "qwen_called": qwen_called,
        "route_source": route_source,
        "target_label": metrics["target_label"],
        "target_defect_type": metrics["target_defect_type"],
        "target_explanation": metrics["target_explanation"],
        "pred_label": pred["pred_label"],
        "pred_defect_type": pred["pred_defect_type"],
        "pred_explanation": pred["pred_explanation"],
        "valid_json": pred["valid_json"],
        "exact_schema": pred["exact_schema"],
        "allowed_type_ok": metrics["allowed_type_ok"],
        "label_correct": metrics["label_correct"],
        "defect_type_exact": metrics["defect_type_exact"],
        "defect_type_hit": metrics["defect_type_hit"],
        "explanation_keyword_hit": metrics["explanation_keyword_hit"],
        "explanation_overlap_count": metrics["explanation_overlap_count"],
        "explanation_target_terms": metrics["explanation_target_terms"],
        "is_true_abnormal": metrics["is_true_abnormal"],
        "has_type_target": metrics["has_type_target"],
        "has_explanation_target": metrics["has_explanation_target"],
        "raw_answer": raw_answer,
        "elapsed_sec": f"{elapsed_sec:.6f}",
    }


def summarize(model_key: str, method: str, dataset: str, rows: list[dict[str, Any]], aggregate) -> dict[str, Any]:
    overall = aggregate(rows)
    qwen_called = sum(int(row["qwen_called"]) for row in rows)
    elapsed = sum(float(row.get("elapsed_sec") or 0.0) for row in rows)
    overall["qwen_called"] = qwen_called
    overall["qwen_rate"] = qwen_called / max(len(rows), 1)
    overall["elapsed_sec_sum"] = elapsed
    overall["sec_per_image"] = elapsed / max(len(rows), 1)
    return {"model_key": model_key, "method": method, "dataset": dataset, **overall}


def radius_by_category(rows: list[dict[str, Any]], eff_scores: dict[tuple[str, str, str], dict[str, Any]], q: float) -> dict[str, float]:
    vals: dict[str, list[float]] = {}
    for row in rows:
        cat = str(row.get("category") or "")
        eff = eff_scores[(str(row["dataset_key"]), "val", str(row["id"]))]
        vals.setdefault(cat, []).append(abs(float(eff["score"]) - float(eff["thr"])))
    return {cat: float(np.quantile(v, q)) if v else 0.0 for cat, v in vals.items()}


def objective(summary: dict[str, Any], target_rate: float) -> float:
    return (
        WEIGHTS["acc"] * float(summary.get("label_accuracy", 0.0))
        + WEIGHTS["type_hit"] * float(summary.get("defect_type_exact_abnormal", 0.0))
        - WEIGHTS["qwen_rate_deviation"] * abs(float(summary.get("qwen_rate", 0.0)) - target_rate)
    )


def evaluate_split(
    rows: list[dict[str, Any]],
    split: str,
    dataset: str,
    model_key: str,
    family: str,
    qpreds: dict[str, dict[str, Any]],
    eff_scores: dict[tuple[str, str, str], dict[str, Any]],
    radii: dict[str, float],
    method: str,
    parse_prediction,
    norm_type,
    row_metrics,
) -> list[dict[str, Any]]:
    final_rows = []
    for row in rows:
        sample_id = str(row["id"])
        eff = eff_scores[(dataset, split, sample_id)]
        eff_pred = int(eff["eff_pred"])
        margin = abs(float(eff["score"]) - float(eff["thr"]))
        radius = radii.get(str(row.get("category") or ""), 0.0)
        called = bool(eff_pred == 1 or margin <= radius)
        if called:
            qrow = qpreds[sample_id]
            pred = qwen_prediction_from_csv(qrow, parse_prediction, norm_type)
            elapsed = float(qrow.get("elapsed_sec") or 0.0)
            raw_answer = qrow.get("raw_answer", "")
            route = "eff_abnormal" if eff_pred == 1 else "gray"
        else:
            pred = deterministic_normal_prediction(dataset)
            elapsed = 0.0
            raw_answer = json.dumps(pred, ensure_ascii=False)
            route = "eff_confident_normal"
        final_rows.append(output_row(row, pred, method, model_key, dataset, int(called), route, elapsed, raw_answer, row_metrics))
    return final_rows


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "eval"))
    from eval_three_field_qwen import aggregate, norm_type, parse_prediction, row_metrics

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    eff_scores = load_eff_scores()
    summary_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    # Pure EfficientAD baseline, model-independent.
    for dataset in DATASETS:
        final_rows = []
        for row in read_jsonl(DATA_ROOT / dataset / "test.jsonl"):
            eff = eff_scores[(dataset, "test", str(row["id"]))]
            eff_pred = int(eff["eff_pred"])
            pred = deterministic_eff_abnormal_prediction(dataset) if eff_pred == 1 else deterministic_normal_prediction(dataset)
            route = "eff_abnormal" if eff_pred == 1 else "eff_normal"
            final_rows.append(output_row(row, pred, "pure_efficientad", "none", dataset, 0, route, 0.0, json.dumps(pred), row_metrics))
        write_csv(RESULT_ROOT / "none" / "pure_efficientad" / f"{dataset}.predictions.csv", final_rows)
        summary_rows.append(summarize("none", "pure_efficientad", dataset, final_rows, aggregate))

    for model_key in MODEL_KEYS:
        for dataset in DATASETS:
            target_rate = REFERENCE_QWEN_RATE[dataset]
            rate_cap = min(GLOBAL_RATE_CAP, target_rate + RATE_CAP_EXTRA)
            val_rows = []
            for row in read_jsonl(DATA_ROOT / dataset / "val.jsonl"):
                row = dict(row)
                row["dataset_key"] = dataset
                val_rows.append(row)
            val_qpreds = load_val_qwen(model_key, dataset, eff_scores)
            best: dict[str, Any] | None = None
            best_radii: dict[str, float] = {}
            for qi in range(0, 101):
                q = qi / 100.0
                radii = radius_by_category(val_rows, eff_scores, q)
                final_rows = evaluate_split(
                    val_rows,
                    "val",
                    dataset,
                    model_key,
                    "ours",
                    val_qpreds,
                    eff_scores,
                    radii,
                    "ours_val",
                    parse_prediction,
                    norm_type,
                    row_metrics,
                )
                summ = summarize(model_key, "ours_val", dataset, final_rows, aggregate)
                obj = objective(summ, target_rate)
                record = {
                    "model_key": model_key,
                    "dataset": dataset,
                    "q": q,
                    "objective": obj,
                    "target_qwen_rate": target_rate,
                    "rate_cap": rate_cap,
                    **summ,
                }
                tuning_rows.append(record)
                if float(summ["qwen_rate"]) <= rate_cap:
                    if best is None or obj > float(best["objective"]) or (
                        abs(obj - float(best["objective"])) < 1e-12 and abs(float(summ["qwen_rate"]) - target_rate) < abs(float(best["qwen_rate"]) - target_rate)
                    ):
                        best = record
                        best_radii = radii
            if best is None:
                raise RuntimeError(f"No candidate under rate cap for {model_key}/{dataset}")

            selected_rows.append(best)
            write_csv(
                RESULT_ROOT / model_key / "selected_radius" / f"{dataset}.csv",
                [{"category": cat, "q": best["q"], "radius": radius} for cat, radius in sorted(best_radii.items())],
            )

            test_qpreds = load_full_qwen(model_key, "ours", dataset)
            test_rows = read_jsonl(DATA_ROOT / dataset / "test.jsonl")
            final_rows = evaluate_split(
                test_rows,
                "test",
                dataset,
                model_key,
                "ours",
                test_qpreds,
                eff_scores,
                best_radii,
                "ours_per_dataset_q",
                parse_prediction,
                norm_type,
                row_metrics,
            )
            write_csv(RESULT_ROOT / model_key / "ours_per_dataset_q" / f"{dataset}.predictions.csv", final_rows)
            summary_rows.append(summarize(model_key, "ours_per_dataset_q", dataset, final_rows, aggregate))

            # Random 40/50/60 baselines with the same Qwen predictions.
            for rate in RANDOM_RATES:
                method = f"random_{int(rate * 100)}"
                final_rows = []
                for row in test_rows:
                    sample_id = str(row["id"])
                    eff = eff_scores[(dataset, "test", sample_id)]
                    eff_pred = int(eff["eff_pred"])
                    u = stable_uniform(dataset, sample_id)
                    direct = u < rate
                    called = bool(direct or (not direct and eff_pred == 1))
                    if called:
                        qrow = test_qpreds[sample_id]
                        pred = qwen_prediction_from_csv(qrow, parse_prediction, norm_type)
                        elapsed = float(qrow.get("elapsed_sec") or 0.0)
                        raw_answer = qrow.get("raw_answer", "")
                    else:
                        pred = deterministic_normal_prediction(dataset)
                        elapsed = 0.0
                        raw_answer = json.dumps(pred, ensure_ascii=False)
                    route = "random_direct_qwen" if direct else ("eff_abnormal_fallback" if eff_pred == 1 else "eff_normal")
                    final_rows.append(output_row(row, pred, method, model_key, dataset, int(called), route, elapsed, raw_answer, row_metrics))
                write_csv(RESULT_ROOT / model_key / method / f"{dataset}.predictions.csv", final_rows)
                summary_rows.append(summarize(model_key, method, dataset, final_rows, aggregate))

    write_csv(RESULT_ROOT / "summary_by_dataset.csv", summary_rows)
    write_csv(RESULT_ROOT / "tuning_candidates.csv", tuning_rows)
    write_csv(RESULT_ROOT / "selected_q_by_dataset.csv", selected_rows)
    (RESULT_ROOT / "objective_config.json").write_text(
        json.dumps(
            {
                "weights": WEIGHTS,
                "reference_qwen_rate": REFERENCE_QWEN_RATE,
                "rate_cap": f"min({GLOBAL_RATE_CAP}, reference + {RATE_CAP_EXTRA})",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"result_root": str(RESULT_ROOT), "summary": str(RESULT_ROOT / "summary_by_dataset.csv")}, indent=2))


if __name__ == "__main__":
    main()
