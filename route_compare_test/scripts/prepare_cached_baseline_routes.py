from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__import__("os").environ["UACR_ROOT"])
DATA_ROOT = ROOT / "data" / "three_field_qwen"
OUT_ROOT = ROOT / "route_compare_test"
EFF_SCORES = ROOT / "data" / "splits" / "hybrid_unified_five" / "merged_val_test_scores.csv"
YOLO_WEIGHTS = OUT_ROOT / "yolo_weights" / "yolo11s-cls.pt"

DATASETS = [
    "goodsad_80p",
    "mvtec_ad_80p",
    "mvtec_loco_80p",
    "visa_80p",
    "ksdd2_mvtlike",
]

CSV_DATASET = {
    "goodsad_80p": "goodsad_80p",
    "mvtec_ad_80p": "mvtec_ad_80p",
    "mvtec_loco_80p": "mvtec_loco_80p",
    "visa_80p": "visa_80p",
    "ksdd2_mvtlike": "ksdd2",
}

RANDOM_SEED = "route_compare_cached_20260629"
RANDOM_RATES = (0.60, 0.70)

# SAEC route follows the official SAEC Scheme-B implementation:
# complexity_split(data_root, target_q_ratio=0.30) computes the complexity
# threshold on the current evaluation split, then sends scores >= threshold to Qwen.
SAEC_HIGH_COMPLEXITY_RATIO = 0.30
YOLO_SMAX_THR = 0.55
YOLO_MARGIN_THR = 0.03
YOLO_ENT_THR = 0.95


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def norm_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"normal", "good", "0"}:
        return "normal"
    if text in {"abnormal", "defect", "defective", "anomaly", "1"}:
        return "abnormal"
    return "unknown"


def route_row_base(row: dict[str, Any], dataset_key: str, method: str) -> dict[str, Any]:
    target = row.get("target") or {}
    return {
        "id": str(row["id"]),
        "dataset_key": dataset_key,
        "dataset": row.get("dataset"),
        "category": row.get("category"),
        "image": row.get("image"),
        "method": method,
        "target_label": norm_label(target.get("label")),
        "target_defect_type": target.get("defect_type", ""),
        "target_explanation": target.get("explanation", ""),
    }


def load_eff_scores() -> dict[tuple[str, str], dict[str, Any]]:
    df = pd.read_csv(EFF_SCORES)
    df = df[df["split2"].astype(str) == "test"].copy()
    return {(str(rec["dataset"]), str(rec["id"])): rec for rec in df.to_dict("records")}


def complexity_score(path: str, size: int = 192) -> float:
    im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        return 0.0
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1)
    entropy = float(-np.sum(p * np.log(p + 1e-12)) / np.log(256))
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_var = float(np.var(lap))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = float(np.mean(np.hypot(gx, gy)))
    edge = float(cv2.Canny(gray, 64, 128).mean())
    enc = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 30])[1]
    im2 = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    residual = cv2.absdiff(im.astype(np.float32), im2.astype(np.float32))
    return float(
        0.30 * entropy
        + 0.25 * edge
        + 0.20 * np.log1p(lap_var) / 8.0
        + 0.15 * (grad / 16.0)
        + 0.10 * (residual.mean() / 255.0)
    )


def score_cache_path(dataset_key: str, split: str) -> Path:
    return OUT_ROOT / "routes_cached" / "complexity" / f"{dataset_key}.{split}.csv"


def complexity_scores(dataset_key: str, split: str, rows: list[dict[str, Any]]) -> dict[str, float]:
    cache = score_cache_path(dataset_key, split)
    if cache.exists():
        with cache.open(newline="", encoding="utf-8") as f:
            cached = {row["id"]: float(row["complexity_score"]) for row in csv.DictReader(f)}
        if len(cached) == len(rows):
            return cached
    out = {str(row["id"]): complexity_score(str(row["image"])) for row in rows}
    write_csv(cache, [{"id": key, "complexity_score": f"{value:.10f}"} for key, value in out.items()])
    return out


def yolo_confidence_rows(rows: list[dict[str, Any]], dataset_key: str) -> dict[str, dict[str, Any]]:
    cache = OUT_ROOT / "routes_cached" / "yolo" / f"{dataset_key}.csv"
    if cache.exists():
        with cache.open(newline="", encoding="utf-8") as f:
            cached = {row["id"]: row for row in csv.DictReader(f)}
        if all(str(row["id"]) in cached for row in rows):
            return cached

    from ultralytics import YOLO

    model = YOLO(str(YOLO_WEIGHTS))
    # Official SAEC runs YOLO as the CPU edge branch.
    device = os.environ.get("YOLO_DEVICE", "cpu")
    out: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows, 1):
        try:
            result = model.predict(str(row["image"]), imgsz=448, device=device, verbose=False)[0]
        except Exception:
            result = model.predict(str(row["image"]), imgsz=448, device="cpu", verbose=False)[0]
        prob = result.probs.data.cpu().numpy()
        sorted_prob = np.sort(prob)[::-1]
        smax = float(sorted_prob[0])
        margin = float(sorted_prob[0] - sorted_prob[1] if sorted_prob.size > 1 else sorted_prob[0])
        ent = float(-(prob * np.log(prob + 1e-12)).sum() / math.log(len(prob)))
        confident_good = bool(smax >= YOLO_SMAX_THR and margin >= YOLO_MARGIN_THR and ent <= YOLO_ENT_THR)
        out[str(row["id"])] = {
            "id": str(row["id"]),
            "yolo_smax": f"{smax:.10f}",
            "yolo_margin": f"{margin:.10f}",
            "yolo_ent": f"{ent:.10f}",
            "yolo_confident_good": int(confident_good),
            "yolo_pred": "good" if confident_good else "defect_or_uncertain",
        }
        if idx % 200 == 0:
            print(f"[yolo] {dataset_key} processed {idx}/{len(rows)}", flush=True)

    write_csv(cache, list(out.values()))
    return out


def main() -> None:
    if not YOLO_WEIGHTS.exists():
        raise SystemExit(f"Missing YOLO weights: {YOLO_WEIGHTS}")

    route_root = OUT_ROOT / "routes_cached"
    route_root.mkdir(parents=True, exist_ok=True)
    eff_by_key = load_eff_scores()
    summary_rows: list[dict[str, Any]] = []

    for dataset_key in DATASETS:
        test_rows = read_jsonl(DATA_ROOT / dataset_key / "test.jsonl")
        csv_dataset = CSV_DATASET[dataset_key]
        print(f"[dataset] {dataset_key} test={len(test_rows)}", flush=True)

        random_routes: dict[str, list[dict[str, Any]]] = {f"random_{int(rate * 100)}": [] for rate in RANDOM_RATES}
        for row in test_rows:
            sample_id = str(row["id"])
            eff = eff_by_key.get((csv_dataset, sample_id))
            if eff is None:
                raise KeyError(f"Missing EfficientAD score for {dataset_key}/{sample_id}")
            eff_pred = int(eff["eff_pred"])
            u = stable_uniform(dataset_key, sample_id)
            for rate in RANDOM_RATES:
                method = f"random_{int(rate * 100)}"
                direct = u < rate
                qwen_called = bool(direct or (not direct and eff_pred == 1))
                route = route_row_base(row, dataset_key, method)
                route.update(
                    {
                        "qwen_called": int(qwen_called),
                        "route_source": "random_direct_qwen" if direct else ("eff_abnormal_fallback" if eff_pred == 1 else "eff_normal"),
                        "random_u": f"{u:.12f}",
                        "eff_score": eff["score"],
                        "eff_threshold": eff["thr"],
                        "eff_pred": eff_pred,
                    }
                )
                random_routes[method].append(route)

        for method, rows in random_routes.items():
            write_csv(route_root / method / f"{dataset_key}.csv", rows)
            summary_rows.append(
                {
                    "method": method,
                    "dataset": dataset_key,
                    "n": len(rows),
                    "qwen_called": sum(int(row["qwen_called"]) for row in rows),
                    "qwen_rate": sum(int(row["qwen_called"]) for row in rows) / max(len(rows), 1),
                }
            )

        test_scores = complexity_scores(dataset_key, "test", test_rows)
        # Official SAEC uses complexity_split(args.data, target_q_ratio=init_q_ratio),
        # so the quantile is computed from the evaluated split itself.
        c_thr = float(np.quantile(list(test_scores.values()), 1.0 - SAEC_HIGH_COMPLEXITY_RATIO))
        low_complexity_rows = [row for row in test_rows if test_scores[str(row["id"])] < c_thr]
        yolo_rows = yolo_confidence_rows(low_complexity_rows, dataset_key)

        saec_routes: list[dict[str, Any]] = []
        for row in test_rows:
            sample_id = str(row["id"])
            score = test_scores[sample_id]
            high_complexity = score >= c_thr
            yr = yolo_rows.get(sample_id, {})
            yolo_good = bool(int(float(yr.get("yolo_confident_good", 0) or 0)))
            qwen_called = bool(high_complexity or not yolo_good)
            route = route_row_base(row, dataset_key, "saec_mod")
            route.update(
                {
                    "qwen_called": int(qwen_called),
                    "route_source": "complexity_qwen" if high_complexity else ("yolo_confident_good" if yolo_good else "yolo_defect_or_uncertain"),
                    "complexity_score": f"{score:.10f}",
                    "complexity_threshold_from_eval_split": f"{c_thr:.10f}",
                    "yolo_smax": yr.get("yolo_smax", ""),
                    "yolo_margin": yr.get("yolo_margin", ""),
                    "yolo_ent": yr.get("yolo_ent", ""),
                    "yolo_pred": yr.get("yolo_pred", "not_run_high_complexity"),
                }
            )
            saec_routes.append(route)

        write_csv(route_root / "saec_mod" / f"{dataset_key}.csv", saec_routes)
        summary_rows.append(
            {
                "method": "saec_mod",
                "dataset": dataset_key,
                "n": len(saec_routes),
                "qwen_called": sum(int(row["qwen_called"]) for row in saec_routes),
                "qwen_rate": sum(int(row["qwen_called"]) for row in saec_routes) / max(len(saec_routes), 1),
                "complexity_threshold_from_eval_split": f"{c_thr:.10f}",
            }
        )

    write_csv(route_root / "route_summary.csv", summary_rows)
    print(f"[done] routes={route_root}", flush=True)


if __name__ == "__main__":
    main()
