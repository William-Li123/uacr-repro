#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__import__("os").environ["UACR_LEGACY_ROOT"])

DATASETS = {
    "mvtec_ad_80p": {
        "eff_dir": "outputs/model_new_eval/efficientad/mvtec_ad_80p",
        "qwen_pred": "outputs/model_new_eval/qwen/mvtec_ad_80p__qwen_mvtec_ad/predictions.csv",
        "qwen_format": "strong_json",
    },
    "visa_80p": {
        "eff_dir": "outputs/model_new_eval/efficientad/visa_80p",
        "qwen_pred": "outputs/model_new_eval/qwen/visa_80p__qwen_visa/predictions.csv",
        "qwen_format": "strong_json",
    },
    "mvtec_loco_80p": {
        "eff_dir": "outputs/model_new_eval/efficientad/mvtec_loco_80p",
        "qwen_pred": "outputs/model_new_eval/qwen/mvtec_loco_80p__qwen_mvtec_loco/predictions.csv",
        "qwen_format": "strong_json",
    },
    "goodsad_80p": {
        "eff_dir": "outputs/goodsad_80p_efficientad_threshold_5001",
        "qwen_pred": "outputs/qwen35_9b_goodsad_80p_sft_strong_json/predictions.csv",
        "qwen_format": "goodsad_strong_json",
    },
    "ksdd2": {
        "eff_dir": "outputs/ksdd2_efficientad_threshold_5001",
        # Use the explanation-capable strong-json KSDD2 run.
        "qwen_pred": "outputs/model_new_eval/qwen/ksdd2__qwen_mvtec_ad/predictions.csv",
        "qwen_format": "strong_json",
    },
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def stable_unit(text: str) -> float:
    value = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)
    return value / float(16**12 - 1)


def best_threshold(labels: np.ndarray, scores: np.ndarray, objective: str = "balanced") -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    thresholds = np.unique(scores)
    if len(thresholds) == 0:
        return 0.0
    best = None
    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        acc = float((pred == labels).mean())
        bal = float(balanced_accuracy_score(labels, pred)) if len(np.unique(labels)) > 1 else acc
        f1 = float(f1_score(labels, pred, zero_division=0)) if len(np.unique(labels)) > 1 else 0.0
        score = {"accuracy": acc, "balanced": bal, "f1": f1}.get(objective, bal)
        rank = (score, acc, bal)
        if best is None or rank > best[0]:
            best = (rank, float(thr))
    return float(best[1])


def split_val_test(df: pd.DataFrame, val_ratio: float, seed: int) -> pd.DataFrame:
    pieces = []
    work = df.copy()
    work["_stratum"] = (
        work["category"].astype(str)
        + "||"
        + work["label_int"].astype(str)
        + "||"
        + work["defect_type"].astype(str)
    )
    for _, g in work.groupby("_stratum", sort=False):
        g = g.copy()
        g["_h"] = g["sample_key"].map(lambda x: stable_unit(f"{seed}:{x}"))
        g = g.sort_values("_h")
        n = len(g)
        n_val = 0 if n <= 1 else max(1, min(n - 1, int(round(n * val_ratio))))
        split = np.array(["test"] * n, dtype=object)
        split[:n_val] = "val"
        g["split2"] = split
        pieces.append(g.drop(columns=["_h"]))
    return pd.concat(pieces, ignore_index=True).drop(columns=["_stratum"])


def load_eff_scores(root: Path, dataset: str) -> pd.DataFrame:
    eff_dir = root / DATASETS[dataset]["eff_dir"]
    frames = []
    for p in sorted(eff_dir.glob("*_scores.csv")):
        df = pd.read_csv(p)
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No EfficientAD scores found for {dataset}: {eff_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"label": "label_int"})
    df["eff_path"] = df["path"].astype(str)
    df["path_key"] = df["eff_path"].map(lambda x: str(Path(x)))
    df["id"] = df["eff_path"].map(lambda x: Path(x).stem)
    df["sample_key"] = df["id"]
    return df[["sample_key", "id", "dataset", "category", "eff_path", "path_key", "defect_type", "label_int", "score"]]


def load_qwen(root: Path, dataset: str) -> pd.DataFrame:
    cfg = DATASETS[dataset]
    p = root / cfg["qwen_pred"]
    df = pd.read_csv(p)
    fmt = cfg["qwen_format"]
    if fmt == "strong_json":
        out = pd.DataFrame()
        out["qwen_id"] = df["id"].astype(str)
        out["qwen_path"] = df["image"].astype(str)
        out["path_key"] = out["qwen_path"].map(lambda x: str(Path(x)))
        out["sample_key"] = out["qwen_id"]
        out["qwen_pred_int"] = np.where(
            df["pred_label"].eq("abnormal"),
            1,
            np.where(df["pred_label"].eq("normal"), 0, np.nan),
        )
        out["qwen_valid_label"] = out["qwen_pred_int"].notna().astype(int)
        out["qwen_type_hit"] = pd.to_numeric(df.get("defect_type_hit", 0), errors="coerce").fillna(0).astype(int)
        out["qwen_type_exact"] = pd.to_numeric(df.get("defect_type_exact", 0), errors="coerce").fillna(0).astype(int)
        out["qwen_valid_json"] = pd.to_numeric(df.get("valid_json", 1), errors="coerce").fillna(1).astype(int)
        out["qwen_answer"] = df.get("answer", "").astype(str)
        out["qwen_elapsed_sec"] = pd.to_numeric(df.get("elapsed_sec", 0), errors="coerce").fillna(0.0)
        return out
    if fmt == "goodsad_strong_json":
        out = pd.DataFrame()
        out["qwen_id"] = df["id"].astype(str)
        out["qwen_path"] = df["image"].astype(str)
        out["path_key"] = out["qwen_path"].map(lambda x: str(Path(x)))
        out["sample_key"] = out["qwen_id"]
        out["qwen_pred_int"] = pd.to_numeric(df["pred"], errors="coerce")
        out["qwen_valid_label"] = out["qwen_pred_int"].notna().astype(int)
        out["qwen_type_hit"] = pd.to_numeric(df.get("defect_correct", 0), errors="coerce").fillna(0).astype(int)
        out["qwen_type_exact"] = out["qwen_type_hit"]
        out["qwen_valid_json"] = 1
        out["qwen_answer"] = df.get("answer", "").astype(str)
        out["qwen_elapsed_sec"] = pd.to_numeric(df.get("elapsed_sec", 0), errors="coerce").fillna(0.0)
        return out
    raise ValueError(fmt)


def merge_eff_qwen(eff: pd.DataFrame, qwen: pd.DataFrame, dataset: str) -> pd.DataFrame:
    # Prefer exact constructed id. KSDD2 strong-json ids include a prefix, so fall back to image path.
    merged = eff.merge(qwen.drop(columns=["path_key"]), left_on="sample_key", right_on="sample_key", how="left")
    missing = merged["qwen_pred_int"].isna()
    if missing.any():
        q2 = qwen.drop(columns=["sample_key"]).drop_duplicates("path_key")
        fill = eff.loc[missing, ["path_key"]].merge(q2, on="path_key", how="left")
        for col in fill.columns:
            if col == "path_key":
                continue
            merged.loc[missing, col] = fill[col].to_numpy()
    still_missing = merged["qwen_pred_int"].isna().sum()
    if still_missing:
        raise RuntimeError(f"{dataset}: {still_missing} rows could not merge Qwen predictions")
    merged["dataset_key"] = dataset
    merged["qwen_pred_int"] = merged["qwen_pred_int"].astype(int)
    return merged


def fit_thresholds(val: pd.DataFrame) -> dict[str, float]:
    thresholds = {}
    for cat, g in val.groupby("category"):
        thresholds[cat] = best_threshold(g["label_int"].to_numpy(), g["score"].to_numpy(), "balanced")
    return thresholds


def add_eff_predictions(df: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    df["thr"] = df["category"].map(thresholds).astype(float)
    df["eff_pred"] = (df["score"] >= df["thr"]).astype(int)
    df["margin"] = (df["score"] - df["thr"]).abs()
    std = df.groupby("category")["score"].transform("std")
    global_std = float(df["score"].std()) or 1.0
    df["score_std"] = std.replace(0, np.nan).fillna(global_std)
    df["norm_margin"] = df["margin"] / df["score_std"]
    df["signed_norm_margin"] = (df["score"] - df["thr"]) / df["score_std"]
    df["eff_correct"] = (df["eff_pred"] == df["label_int"]).astype(int)
    df["qwen_correct"] = (df["qwen_pred_int"] == df["label_int"]).astype(int)
    df["qwen_benefit"] = df["qwen_correct"] - df["eff_correct"]
    return df


def qwen_sec(df: pd.DataFrame) -> float:
    sec = pd.to_numeric(df["qwen_elapsed_sec"], errors="coerce").fillna(0)
    if float(sec.mean()) > 0:
        return float(sec.mean())
    return 1.0


def eff_sec(dataset: str) -> float:
    # Measured previously for model_new, and summary-derived rough values for GoodsAD/KSDD2.
    return {
        "mvtec_ad_80p": 0.022,
        "visa_80p": 0.007,
        "mvtec_loco_80p": 0.024,
        "goodsad_80p": 0.020,
        "ksdd2": 0.018,
    }.get(dataset, 0.02)


def compute_metrics(df: pd.DataFrame, route_decision: np.ndarray, explain_extra: np.ndarray) -> dict[str, Any]:
    route_decision = np.asarray(route_decision, dtype=bool)
    explain_extra = np.asarray(explain_extra, dtype=bool) & ~route_decision
    qwen_called = route_decision | explain_extra
    final_pred = np.where(route_decision, df["qwen_pred_int"].to_numpy(), df["eff_pred"].to_numpy()).astype(int)
    y = df["label_int"].to_numpy().astype(int)
    acc = float((final_pred == y).mean())
    bal = float(balanced_accuracy_score(y, final_pred)) if len(np.unique(y)) > 1 else acc
    final_abn = final_pred == 1
    true_abn = y == 1
    qs = qwen_sec(df)
    avg_sec = eff_sec(str(df["dataset_key"].iloc[0])) + float(qwen_called.mean()) * qs
    return {
        "n": int(len(df)),
        "accuracy": acc,
        "balanced_accuracy": bal,
        "decision_qwen_rate": float(route_decision.mean()),
        "total_qwen_rate": float(qwen_called.mean()),
        "final_abnormal_explanation_coverage": float(qwen_called[final_abn].mean()) if final_abn.any() else np.nan,
        "true_abnormal_explanation_coverage": float(qwen_called[true_abn].mean()) if true_abn.any() else np.nan,
        "qwen_type_hit_on_called": float(df.loc[qwen_called, "qwen_type_hit"].mean()) if qwen_called.any() else np.nan,
        "sec_per_image": avg_sec,
        "qwen_sec_per_image": qs,
        "speedup_vs_qwen": qs / avg_sec if avg_sec > 0 else np.nan,
    }


@dataclass
class Candidate:
    dataset: str
    method: str
    params: dict[str, Any]
    val: dict[str, Any]
    test: dict[str, Any]


def one_hot_features(train: pd.DataFrame, other: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    num_cols = ["score", "thr", "margin", "norm_margin", "signed_norm_margin", "eff_pred"]
    try:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
    xt = enc.fit_transform(train[["category"]])
    xo = enc.transform(other[["category"]])
    return np.hstack([train[num_cols].to_numpy(float), xt]), np.hstack([other[num_cols].to_numpy(float), xo])


def eval_candidates(dataset: str, val: pd.DataFrame, test: pd.DataFrame) -> list[Candidate]:
    out: list[Candidate] = []
    n0 = np.zeros(len(val), dtype=bool)
    n0t = np.zeros(len(test), dtype=bool)
    out.append(Candidate(dataset, "efficientad_only_val_threshold", {}, compute_metrics(val, n0, n0), compute_metrics(test, n0t, n0t)))
    allv = np.ones(len(val), dtype=bool)
    allt = np.ones(len(test), dtype=bool)
    out.append(Candidate(dataset, "qwen_all", {}, compute_metrics(val, allv, n0), compute_metrics(test, allt, n0t)))

    # Unified interpretable rule: Qwen verifies EfficientAD abnormal predictions and gray-zone samples;
    # Qwen is also called explanation-only for any final abnormal not already verified.
    for q in np.round(np.linspace(0.02, 0.80, 40), 3):
        radius = val.groupby("category")["norm_margin"].quantile(q).to_dict()
        v_gray = val.apply(lambda r: r["norm_margin"] <= radius.get(r["category"], 0.0), axis=1).to_numpy()
        t_gray = test.apply(lambda r: r["norm_margin"] <= radius.get(r["category"], 0.0), axis=1).to_numpy()
        v_route = val["eff_pred"].eq(1).to_numpy() | v_gray
        t_route = test["eff_pred"].eq(1).to_numpy() | t_gray
        v_explain = val["eff_pred"].eq(1).to_numpy() & ~v_route
        t_explain = test["eff_pred"].eq(1).to_numpy() & ~t_route
        out.append(Candidate(dataset, "unified_abnormal_or_gray", {"gray_quantile": float(q)}, compute_metrics(val, v_route, v_explain), compute_metrics(test, t_route, t_explain)))

    # Unified learned router: predict validation benefit of asking Qwen from EfficientAD-only features.
    x_val, x_test = one_hot_features(val, test)
    target = val["qwen_benefit"].to_numpy(float)
    if len(np.unique(target)) > 1:
        model = GradientBoostingRegressor(random_state=42, n_estimators=160, max_depth=3, learning_rate=0.04)
        model.fit(x_val, target)
        s_val = model.predict(x_val)
        s_test = model.predict(x_test)
        for budget in np.round(np.linspace(0.02, 0.80, 40), 3):
            kv = int(round(len(val) * budget))
            kt = int(round(len(test) * budget))
            v_route = np.zeros(len(val), dtype=bool)
            t_route = np.zeros(len(test), dtype=bool)
            if kv > 0:
                idx = np.argsort(-s_val)[:kv]
                v_route[idx] = s_val[idx] > 0
            if kt > 0:
                idx = np.argsort(-s_test)[:kt]
                t_route[idx] = s_test[idx] > 0
            # Explanation rule is part of the unified system: if the final decision is an EfficientAD abnormal
            # and Qwen was not used for verification, call Qwen only to explain.
            v_explain = val["eff_pred"].eq(1).to_numpy() & ~v_route
            t_explain = test["eff_pred"].eq(1).to_numpy() & ~t_route
            out.append(Candidate(dataset, "unified_learned_utility", {"budget": float(budget)}, compute_metrics(val, v_route, v_explain), compute_metrics(test, t_route, t_explain)))
    return out


def rows_from_candidates(cands: list[Candidate]) -> pd.DataFrame:
    rows = []
    for c in cands:
        row = {"dataset": c.dataset, "method": c.method, "params": json.dumps(c.params, sort_keys=True, ensure_ascii=False)}
        for split, metrics in [("val", c.val), ("test", c.test)]:
            for k, v in metrics.items():
                row[f"{split}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def select_unified(rows: pd.DataFrame) -> pd.DataFrame:
    chosen = []
    for ds, g in rows.groupby("dataset"):
        eff = g[g["method"].eq("efficientad_only_val_threshold")].iloc[0]
        qwen = g[g["method"].eq("qwen_all")].iloc[0]
        # Same objective for every dataset: improve over EfficientAD, keep explanation coverage,
        # and penalize Qwen calls. Use validation only for selection.
        cand = g[g["method"].isin(["unified_abnormal_or_gray", "unified_learned_utility"])].copy()
        cand["meets_eff"] = cand["val_accuracy"] >= float(eff["val_accuracy"])
        cand["objective"] = (
            cand["val_accuracy"]
            + 0.04 * cand["val_balanced_accuracy"]
            + 0.03 * cand["val_final_abnormal_explanation_coverage"].fillna(0)
            - 0.055 * cand["val_total_qwen_rate"]
        )
        # Prefer candidates that beat EfficientAD on val. If none do, still pick max objective.
        pool = cand[cand["meets_eff"]].copy()
        if pool.empty:
            pool = cand
        chosen.append(pool.sort_values("objective", ascending=False).iloc[0])
    return pd.DataFrame(chosen)


def pct(x: float) -> str:
    return f"{100 * float(x):.2f}%"


def write_report(out_dir: Path, all_rows: pd.DataFrame, selected: pd.DataFrame) -> None:
    lines = [
        "# Unified Five-Dataset Hybrid Routing",
        "",
        "Unified method selection used the same validation objective for every dataset.",
        "Each dataset was split from the existing constructed test set into stratified validation and holdout test.",
        "",
        "## Selected Unified Router",
        "| dataset | method | params | acc | bal_acc | qwen_rate | decision_rate | explain_final_abn | sec/img | qwen_only_sec | speedup_vs_qwen | type_hit_called |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in selected.sort_values("dataset").iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["dataset"]),
                    str(r["method"]),
                    str(r["params"]),
                    pct(r["test_accuracy"]),
                    pct(r["test_balanced_accuracy"]),
                    pct(r["test_total_qwen_rate"]),
                    pct(r["test_decision_qwen_rate"]),
                    pct(r["test_final_abnormal_explanation_coverage"]) if pd.notna(r["test_final_abnormal_explanation_coverage"]) else "",
                    f"{float(r['test_sec_per_image']):.3f}",
                    f"{float(r['test_qwen_sec_per_image']):.3f}",
                    f"{float(r['test_speedup_vs_qwen']):.2f}x",
                    pct(r["test_qwen_type_hit_on_called"]) if pd.notna(r["test_qwen_type_hit_on_called"]) else "",
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Baselines",
        "| dataset | baseline | acc | bal_acc | qwen_rate | sec/img | speedup_vs_qwen | type_hit_called |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    base = all_rows[all_rows["method"].isin(["efficientad_only_val_threshold", "qwen_all"])].copy()
    for _, r in base.sort_values(["dataset", "method"]).iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["dataset"]),
                    str(r["method"]),
                    pct(r["test_accuracy"]),
                    pct(r["test_balanced_accuracy"]),
                    pct(r["test_total_qwen_rate"]),
                    f"{float(r['test_sec_per_image']):.3f}",
                    f"{float(r['test_speedup_vs_qwen']):.2f}x",
                    pct(r["test_qwen_type_hit_on_called"]) if pd.notna(r["test_qwen_type_hit_on_called"]) else "",
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Overall Route",
        "",
        "1. Run EfficientAD for every image and obtain an anomaly score.",
        "2. Use validation labels to fit one category-specific EfficientAD threshold.",
        "3. Compute normalized margin `abs(score - threshold) / category_score_std`.",
        "4. The selected unified router decides whether Qwen should verify the label:",
        "   - `unified_abnormal_or_gray`: verify if EfficientAD predicts abnormal or the normalized margin is in the validation-calibrated gray zone.",
        "   - `unified_learned_utility`: learn from validation whether Qwen is likely to improve over EfficientAD using only EfficientAD score, margin, signed margin, EfficientAD prediction, and category.",
        "5. If Qwen verifies, final label/type/explanation come from Qwen, with EfficientAD score retained as supporting evidence.",
        "6. If Qwen is not used for verification and EfficientAD final label is abnormal, call Qwen explanation-only so final abnormal outputs still have text evidence.",
        "7. If EfficientAD final label is normal and not in the route, skip Qwen entirely.",
        "",
        "The method is unified because the feature set, validation objective, and routing/explanation rules are the same for all five datasets. Dataset-specific prompts/adapters only affect the Qwen call itself.",
    ]
    (out_dir / "unified_five_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--out-dir", default=str(ROOT / "hybrid_unified_five"))
    parser.add_argument("--val-ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=20260623)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    merged_frames = []
    threshold_dump = {}
    for ds in DATASETS:
        eff = load_eff_scores(root, ds)
        qwen = load_qwen(root, ds)
        data = merge_eff_qwen(eff, qwen, ds)
        data = split_val_test(data, args.val_ratio, args.seed)
        val0 = data[data["split2"].eq("val")].copy()
        test0 = data[data["split2"].eq("test")].copy()
        thr = fit_thresholds(val0)
        threshold_dump[ds] = thr
        val = add_eff_predictions(val0, thr)
        test = add_eff_predictions(test0, thr)
        val["split2"] = "val"
        test["split2"] = "test"
        merged_frames.append(pd.concat([val, test], ignore_index=True))
        candidates.extend(eval_candidates(ds, val, test))

    rows = rows_from_candidates(candidates)
    rows.to_csv(out_dir / "all_candidates.csv", index=False)
    merged = pd.concat(merged_frames, ignore_index=True)
    merged.to_csv(out_dir / "merged_val_test_scores.csv", index=False)
    selected = select_unified(rows)
    selected.to_csv(out_dir / "selected_unified_router.csv", index=False)
    (out_dir / "thresholds.json").write_text(json.dumps(threshold_dump, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out_dir, rows, selected)
    print(out_dir / "unified_five_report.md")


if __name__ == "__main__":
    main()
