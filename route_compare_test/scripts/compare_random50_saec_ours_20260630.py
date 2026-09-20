from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__import__("os").environ["UACR_ROOT"])
DATA_ROOT = ROOT / "data" / "three_field_qwen"
OUT_ROOT = ROOT / "route_compare_test"
RESULT_ROOT = OUT_ROOT / "compare_20260630"
EFF_SCORES = ROOT / "data" / "splits" / "hybrid_unified_five" / "merged_val_test_scores.csv"

DATASETS = ["goodsad_80p", "mvtec_ad_80p", "mvtec_loco_80p", "visa_80p", "ksdd2_mvtlike"]
TUNE_DATASETS = ["goodsad_80p", "mvtec_ad_80p", "mvtec_loco_80p", "visa_80p"]
MODEL_KEYS = ["qwen35_9b", "qwen25vl_7b"]
RANDOM_RATES = [0.50, 0.60]
RANDOM_SEED = "route_compare_cached_20260630"

CSV_DATASET_TO_KEY = {
    "goodsad_80p": "goodsad_80p",
    "mvtec_ad_80p": "mvtec_ad_80p",
    "mvtec_loco_80p": "mvtec_loco_80p",
    "visa_80p": "visa_80p",
    "ksdd2": "ksdd2_mvtlike",
}


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
        if not dataset_key:
            continue
        out[(dataset_key, str(rec["split2"]), str(rec["id"]))] = rec
    return out


def load_full_qwen(model_key: str, family: str, dataset: str) -> dict[str, dict[str, Any]]:
    path = OUT_ROOT / "qwen_full_test_outputs" / model_key / family / dataset / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return {str(row["id"]): row for row in read_csv(path)}


def load_val_qwen(model_key: str, dataset: str) -> dict[str, dict[str, Any]]:
    path = ROOT / "model_qwen_adapter" / "sft_lora" / model_key / dataset / "val_eval" / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return {str(row["id"]): row for row in read_csv(path)}


def deterministic_normal_prediction(dataset: str) -> dict[str, Any]:
    if dataset == "ksdd2_mvtlike":
        return {
            "valid_json": 1,
            "exact_schema": 1,
            "pred_label": "normal",
            "pred_defect_type": "",
            "pred_explanation": "",
        }
    return {
        "valid_json": 1,
        "exact_schema": 1,
        "pred_label": "normal",
        "pred_defect_type": "good",
        "pred_explanation": "The query matches the normal references and shows no visible defect.",
    }


def deterministic_eff_abnormal_prediction(dataset: str) -> dict[str, Any]:
    return {
        "valid_json": 1,
        "exact_schema": 1,
        "pred_label": "abnormal",
        "pred_defect_type": "",
        "pred_explanation": "",
    }


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


def category_radius(rows: list[dict[str, Any]], eff_scores: dict[tuple[str, str, str], dict[str, Any]], q: float) -> dict[tuple[str, str], float]:
    by_cat: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        dataset = str(row["dataset_key"])
        cat = str(row.get("category") or "")
        eff = eff_scores[(dataset, "val", str(row["id"]))]
        by_cat.setdefault((dataset, cat), []).append(abs(float(eff["score"]) - float(eff["thr"])))
    return {key: float(np.quantile(vals, q)) if vals else 0.0 for key, vals in by_cat.items()}


def make_val_rows_for_q(
    model_key: str,
    q: float,
    eff_scores: dict[tuple[str, str, str], dict[str, Any]],
    parse_prediction,
    norm_type,
    row_metrics,
    aggregate,
) -> tuple[dict[str, Any], dict[tuple[str, str], float]]:
    val_manifests: list[dict[str, Any]] = []
    for dataset in TUNE_DATASETS:
        for row in read_jsonl(DATA_ROOT / dataset / "val.jsonl"):
            row = dict(row)
            row["dataset_key"] = dataset
            val_manifests.append(row)
    radii = category_radius(val_manifests, eff_scores, q)

    rows_out: list[dict[str, Any]] = []
    for dataset in TUNE_DATASETS:
        qpreds = load_val_qwen(model_key, dataset)
        for row in read_jsonl(DATA_ROOT / dataset / "val.jsonl"):
            sample_id = str(row["id"])
            eff = eff_scores[(dataset, "val", sample_id)]
            eff_pred = int(eff["eff_pred"])
            margin = abs(float(eff["score"]) - float(eff["thr"]))
            radius = radii[(dataset, str(row.get("category") or ""))]
            called = bool(eff_pred == 1 or margin <= radius)
            if called:
                qrow = qpreds[sample_id]
                pred = qwen_prediction_from_csv(qrow, parse_prediction, norm_type)
                elapsed = float(qrow.get("elapsed_sec") or 0.0)
                raw_answer = qrow.get("raw_answer", "")
                route_source = "eff_abnormal" if eff_pred == 1 else "gray"
            else:
                pred = deterministic_normal_prediction(dataset)
                elapsed = 0.0
                raw_answer = json.dumps(pred, ensure_ascii=False)
                route_source = "eff_confident_normal"
            rows_out.append(output_row(row, pred, "ours_tuned_val", model_key, dataset, int(called), route_source, elapsed, raw_answer, row_metrics))

    overall = aggregate(rows_out)
    called = sum(int(row["qwen_called"]) for row in rows_out)
    overall["qwen_called"] = called
    overall["qwen_rate"] = called / max(len(rows_out), 1)
    return overall, radii


def objective(row: dict[str, Any]) -> float:
    # Explanation-first Pareto objective. QwenRate is a hard constraint in tune_q
    # (<= 0.70), so the soft penalty is intentionally mild.
    return (
        0.03 * float(row.get("label_accuracy", 0.0))
        + 0.05 * float(row.get("balanced_accuracy", 0.0))
        + 0.20 * float(row.get("defect_type_exact_abnormal", 0.0))
        + 0.20 * float(row.get("explanation_keyword_hit_abnormal", 0.0))
        - 0.005 * float(row.get("qwen_rate", 0.0))
    )


def tune_q(model_key: str, eff_scores: dict[tuple[str, str, str], dict[str, Any]], parse_prediction, norm_type, row_metrics, aggregate) -> tuple[float, dict[tuple[str, str], float], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    best_q = 0.0
    best_radii: dict[tuple[str, str], float] = {}
    best_obj = -1e9
    for q_i in range(0, 71):
        q = q_i / 100.0
        overall, radii = make_val_rows_for_q(model_key, q, eff_scores, parse_prediction, norm_type, row_metrics, aggregate)
        obj = objective(overall)
        record = {"model_key": model_key, "q": q, "objective": obj, **overall}
        candidates.append(record)
        if float(overall["qwen_rate"]) <= 0.70 and (obj > best_obj or (abs(obj - best_obj) < 1e-9 and q < best_q)):
            best_obj = obj
            best_q = q
            best_radii = radii
    return best_q, best_radii, candidates


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "eval"))
    from eval_three_field_qwen import aggregate, norm_type, parse_prediction, row_metrics

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    eff_scores = load_eff_scores()
    all_summaries: list[dict[str, Any]] = []
    all_weighted: dict[tuple[str, str], list[dict[str, Any]]] = {}
    tune_rows: list[dict[str, Any]] = []

    # Build pure EfficientAD rows once per dataset. These are model-independent.
    pure_eff_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset in DATASETS:
        final_rows: list[dict[str, Any]] = []
        for row in read_jsonl(DATA_ROOT / dataset / "test.jsonl"):
            eff = eff_scores[(dataset, "test", str(row["id"]))]
            eff_pred = int(eff["eff_pred"])
            if eff_pred == 1:
                pred = deterministic_eff_abnormal_prediction(dataset)
                route = "eff_abnormal"
            else:
                pred = deterministic_normal_prediction(dataset)
                route = "eff_normal"
            final_rows.append(output_row(row, pred, "pure_efficientad", "none", dataset, 0, route, 0.0, json.dumps(pred), row_metrics))
        pure_eff_by_dataset[dataset] = final_rows
        write_csv(RESULT_ROOT / "pure_efficientad" / f"{dataset}.predictions.csv", final_rows)
        all_summaries.append(summarize("none", "pure_efficientad", dataset, final_rows, aggregate))
        all_weighted.setdefault(("none", "pure_efficientad"), []).extend(final_rows)

    for model_key in MODEL_KEYS:
        best_q, radii, candidates = tune_q(model_key, eff_scores, parse_prediction, norm_type, row_metrics, aggregate)
        tune_rows.extend(candidates)
        radius_rows = [
            {"model_key": model_key, "dataset": key[0], "category": key[1], "q": best_q, "radius": value}
            for key, value in sorted(radii.items())
        ]
        write_csv(RESULT_ROOT / model_key / "ours_tuned_radius.csv", radius_rows)

        for dataset in DATASETS:
            manifest_rows = read_jsonl(DATA_ROOT / dataset / "test.jsonl")
            qpred_ours = load_full_qwen(model_key, "ours", dataset)
            qpred_saec = load_full_qwen(model_key, "saec", dataset)

            for rate in RANDOM_RATES:
                method = f"random_{int(rate * 100)}"
                final_rows: list[dict[str, Any]] = []
                for row in manifest_rows:
                    sample_id = str(row["id"])
                    eff = eff_scores[(dataset, "test", sample_id)]
                    eff_pred = int(eff["eff_pred"])
                    u = stable_uniform(dataset, sample_id)
                    direct = u < rate
                    called = bool(direct or (not direct and eff_pred == 1))
                    if called:
                        qrow = qpred_ours[sample_id]
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
                all_summaries.append(summarize(model_key, method, dataset, final_rows, aggregate))
                all_weighted.setdefault((model_key, method), []).extend(final_rows)

            # SAEC route uses existing cached route and the SAEC adapter Qwen predictions.
            saec_routes = {str(row["id"]): row for row in read_csv(OUT_ROOT / "routes_cached" / "saec_mod" / f"{dataset}.csv")}
            final_rows = []
            for row in manifest_rows:
                sample_id = str(row["id"])
                route_row = saec_routes[sample_id]
                called = int(float(route_row.get("qwen_called", 0) or 0))
                if called:
                    qrow = qpred_saec[sample_id]
                    pred = qwen_prediction_from_csv(qrow, parse_prediction, norm_type)
                    elapsed = float(qrow.get("elapsed_sec") or 0.0)
                    raw_answer = qrow.get("raw_answer", "")
                else:
                    pred = deterministic_normal_prediction(dataset)
                    elapsed = 0.0
                    raw_answer = json.dumps(pred, ensure_ascii=False)
                final_rows.append(output_row(row, pred, "saec_mod", model_key, dataset, called, route_row.get("route_source", ""), elapsed, raw_answer, row_metrics))
            write_csv(RESULT_ROOT / model_key / "saec_mod" / f"{dataset}.predictions.csv", final_rows)
            all_summaries.append(summarize(model_key, "saec_mod", dataset, final_rows, aggregate))
            all_weighted.setdefault((model_key, "saec_mod"), []).extend(final_rows)

            # Ours tuned route.
            final_rows = []
            for row in manifest_rows:
                sample_id = str(row["id"])
                eff = eff_scores[(dataset, "test", sample_id)]
                eff_pred = int(eff["eff_pred"])
                key = (dataset, str(row.get("category") or ""))
                radius = radii.get(key)
                if radius is None:
                    radius = float(np.quantile([abs(float(eff_scores[(dataset, "test", str(r["id"]))]["score"]) - float(eff_scores[(dataset, "test", str(r["id"]))]["thr"])) for r in manifest_rows], best_q))
                margin = abs(float(eff["score"]) - float(eff["thr"]))
                called = bool(eff_pred == 1 or margin <= radius)
                if called:
                    qrow = qpred_ours[sample_id]
                    pred = qwen_prediction_from_csv(qrow, parse_prediction, norm_type)
                    elapsed = float(qrow.get("elapsed_sec") or 0.0)
                    raw_answer = qrow.get("raw_answer", "")
                    route = "eff_abnormal" if eff_pred == 1 else "gray"
                else:
                    pred = deterministic_normal_prediction(dataset)
                    elapsed = 0.0
                    raw_answer = json.dumps(pred, ensure_ascii=False)
                    route = "eff_confident_normal"
                final_rows.append(output_row(row, pred, "ours_tuned", model_key, dataset, int(called), route, elapsed, raw_answer, row_metrics))
            write_csv(RESULT_ROOT / model_key / "ours_tuned" / f"{dataset}.predictions.csv", final_rows)
            all_summaries.append(summarize(model_key, "ours_tuned", dataset, final_rows, aggregate))
            all_weighted.setdefault((model_key, "ours_tuned"), []).extend(final_rows)

    weighted_rows = [summarize(model_key, method, "ALL_WEIGHTED", rows, aggregate) for (model_key, method), rows in sorted(all_weighted.items())]
    write_csv(RESULT_ROOT / "summary_by_dataset.csv", all_summaries)
    write_csv(RESULT_ROOT / "summary_weighted.csv", weighted_rows)
    write_csv(RESULT_ROOT / "tuning_candidates.csv", tune_rows)
    print(json.dumps({"result_root": str(RESULT_ROOT), "summary_weighted": str(RESULT_ROOT / "summary_weighted.csv")}, indent=2))


if __name__ == "__main__":
    main()
