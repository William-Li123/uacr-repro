from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__import__("os").environ["UACR_ROOT"])
SRC_QWEN = ROOT / "data" / "data_qwen"
SRC_ADAPTERS = ROOT / "data" / "data_adapters"
MANIFEST_ROOT = ROOT / "data" / "splits" / "typehit_optimization_v5_full_short_loco" / "manifests"
OUT_ROOT = ROOT / "data" / "three_field_qwen"

DATASETS = {
    "goodsad_80p": {"name": "GoodsAD", "manifest": "goodsad_80p.jsonl"},
    "mvtec_ad_80p": {"name": "MVTec AD", "manifest": "mvtec_ad_80p.jsonl"},
    "visa_80p": {"name": "VisA", "manifest": "visa_80p.jsonl"},
    "mvtec_loco_80p": {"name": "MVTec LOCO", "manifest": "mvtec_loco_80p.jsonl"},
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
NORMAL_EXPL = "The query matches the normal references and shows no visible defect."
LOCO_LOGICAL = (
    "The query violates the normal logical arrangement, count, identity, "
    "or spatial relationship of parts compared with the references."
)
LOCO_STRUCTURAL = (
    "The query has a local structural or surface appearance defect that differs from the normal references."
)

GOODSAD_DEFECT_EXPL = {
    "opened": "The package is opened or unsealed compared with the normal closed references.",
    "cap_open": "The cap is open instead of closed as in the normal references.",
    "cap_half_open": "The cap is partially open instead of fully closed as in the normal references.",
    "straw_missing": "The expected straw or accessory is missing compared with normal examples.",
    "deformation": "The object shape is bent, dented, crushed, or deformed compared with normal examples.",
    "broken": "The package material is broken, torn, or has a visible hole.",
    "surface_damage": "The surface shows visible damage, scratches, stains, or contamination.",
    "surface_anomaly": "The package surface shows abnormal marks, texture, holes, or contamination.",
}


def clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def limit_sentence(text: str, max_chars: int = 260) -> str:
    text = clean_space(text)
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    chosen = ""
    for part in parts:
        part = clean_space(part)
        if not part:
            continue
        if not chosen:
            chosen = part
        elif len(chosen) + 1 + len(part) <= max_chars:
            chosen += " " + part
        else:
            break
    if len(chosen) > max_chars:
        chosen = chosen[:max_chars].rstrip(" ,;:") + "."
    if chosen and chosen[-1] not in ".!?":
        chosen += "."
    return chosen


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_response_explanation(row: dict) -> str:
    try:
        obj = json.loads(row.get("response") or "")
    except Exception:
        return ""
    return clean_space(obj.get("explanation") or obj.get("visual_evidence") or obj.get("evidence") or "")


def read_goodsad_text(row: dict) -> str:
    path = row.get("text_path") or ""
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="ignore")
    paras = [clean_space(x) for x in re.split(r"\n\s*\n", text) if clean_space(x)]
    for para in paras[::-1]:
        low = para.lower()
        if any(k in low for k in ["defect", "anomaly", "opened", "tampered", "missing", "damage", "deform", "scratch", "stain", "broken"]):
            return para
    return paras[-1] if paras else ""


def goodsad_specific_sentence(text: str, defect_type: str) -> str:
    text = clean_space(text)
    if not text:
        return ""
    sentences = [clean_space(x) for x in re.split(r"(?<=[.!?])\s+", text) if clean_space(x)]
    defect_terms = {
        "opened": ["opened", "open", "unsealed", "tampered", "seal", "flap"],
        "cap_open": ["cap", "open"],
        "cap_half_open": ["cap", "half", "partially"],
        "straw_missing": ["straw", "missing", "accessory"],
        "deformation": ["deformed", "deformation", "bent", "dented", "crushed", "shape"],
        "broken": ["broken", "torn", "hole", "break"],
        "surface_damage": ["damage", "scratch", "stain", "contamination", "surface"],
        "surface_anomaly": ["mark", "texture", "surface", "hole", "contamination", "anomaly"],
    }.get(defect_type, [defect_type.replace("_", " ")])
    best_idx, best_score = 0, -10
    for idx, sent in enumerate(sentences):
        low = sent.lower()
        score = 3 * sum(term in low for term in defect_terms)
        score += 2 * sum(term in low for term in ["located", "where", "visible", "appears", "compared"])
        score += 1 * sum(term in low for term in ["top", "bottom", "left", "right", "center", "surface", "package"])
        score -= 3 * sum(term in low for term in ["apparent defect", "there is a defect", "there is an apparent defect"])
        if score > best_score:
            best_idx, best_score = idx, score
    if not sentences:
        return ""
    chosen = sentences[best_idx]
    if best_idx + 1 < len(sentences) and len(chosen) < 180:
        nxt = sentences[best_idx + 1]
        if any(term in nxt.lower() for term in defect_terms):
            chosen += " " + nxt
    return chosen


def normalize_defect_type(row: dict, dataset_key: str) -> str:
    if dataset_key == "ksdd2_mvtlike":
        return ""
    label = str(row.get("label", "")).lower()
    defect_type = str(row.get("defect_type") or "good").strip()
    if label == "normal":
        return "good"
    if dataset_key == "mvtec_loco_80p":
        family = str(row.get("anomaly_family") or "").lower()
        if "logical" in defect_type or family == "logical":
            return "logical_anomaly"
        if "structural" in defect_type or family == "structural":
            return "structural_anomaly"
        return "structural_anomaly"
    return defect_type or "unknown_defect"


def explanation_for(row: dict, dataset_key: str, label: str, defect_type: str) -> str:
    if dataset_key == "ksdd2_mvtlike":
        return ""
    category = row.get("category", "product")
    if label == "normal" or defect_type == "good":
        return NORMAL_EXPL
    if dataset_key == "goodsad_80p":
        text = read_goodsad_text(row) or parse_response_explanation(row) or GOODSAD_DEFECT_EXPL.get(defect_type, "")
        fallback = f"The {category} shows a visible {defect_type.replace('_', ' ')} anomaly compared with normal references."
        return limit_sentence(goodsad_specific_sentence(text, defect_type) or text or fallback)
    if dataset_key == "mvtec_ad_80p":
        defect = clean_space(row.get("qa_defect_type_text")) or defect_type.replace("_", " ")
        location = clean_space(row.get("qa_location_text"))
        appearance = clean_space(row.get("qa_appearance_text"))
        effect = clean_space(row.get("qa_effect_text"))
        parts = [f"The query shows {defect.rstrip('.').lower()}."]
        if location and location.lower() not in {"none", "n/a", "unknown"}:
            parts.append(f"It is located {location.rstrip('.').lower()}.")
        if appearance and appearance.lower() not in {"none", "n/a", "unknown"}:
            parts.append(f"The visual cue is {appearance.rstrip('.').lower()}.")
        elif effect and effect.lower() not in {"none", "n/a", "unknown"}:
            parts.append(effect.rstrip(".") + ".")
        return limit_sentence(" ".join(parts))
    if dataset_key == "visa_80p":
        appearance = clean_space(row.get("qa_appearance_text"))
        defect = clean_space(row.get("qa_defect_type_text")) or ", ".join(row.get("defect_types") or []) or defect_type.replace("_", " ")
        return limit_sentence(f"The query shows {(appearance or defect).rstrip('.').lower()} compared with the normal references.")
    if dataset_key == "mvtec_loco_80p":
        return LOCO_LOGICAL if defect_type == "logical_anomaly" else LOCO_STRUCTURAL
    return limit_sentence(parse_response_explanation(row))


def allowed_types_for(rows: list[dict], dataset_key: str, category: str) -> list[str]:
    if dataset_key == "ksdd2_mvtlike":
        return []
    if dataset_key == "mvtec_loco_80p":
        return ["good", "logical_anomaly", "structural_anomaly"]
    vals = {"good"}
    for row in rows:
        if row.get("category") != category:
            continue
        vals.update(str(x) for x in (row.get("allowed_defect_types") or []) if x)
        vals.update(str(x) for x in (row.get("defect_types") or []) if x)
        if row.get("defect_type"):
            vals.add(str(row["defect_type"]))
    return sorted(vals)


def make_prompt(dataset_key: str, dataset_name: str, category: str, ref_count: int, allowed: list[str]) -> str:
    if dataset_key == "ksdd2_mvtlike":
        return (
            "You are a careful industrial visual anomaly inspector. Dataset: KSDD2. Category: steel surface. "
            "The first image(s), if present, are normal reference examples; the last image is the query image. "
            'Return only one compact JSON object with exactly three keys: "label", "defect_type", and "explanation". '
            'The label must be exactly "normal" or "abnormal". For KSDD2 here, "defect_type" and "explanation" '
            "must be empty strings because no reliable fine-grained textual labels are available."
        )
    extra = ""
    if dataset_key == "mvtec_loco_80p":
        extra = (
            " MVTec LOCO uses coarse defect_type labels: logical_anomaly means wrong count, missing or extra object, "
            "wrong object, or wrong arrangement; structural_anomaly means local damage, contamination, deformation, "
            "or surface/appearance change."
        )
    return (
        f"You are a careful industrial visual anomaly inspector. Dataset: {dataset_name}. Category: {category}. "
        f"The first {ref_count} image(s) are normal reference examples from the same category; the last image is the query image. "
        "Compare the query with the references and decide whether it is normal or abnormal. "
        f"Allowed defect_type values are: {', '.join(allowed)}. "
        'Return only one compact JSON object with exactly three keys: "label", "defect_type", and "explanation". '
        'The label must be exactly "normal" or "abnormal". For normal images, defect_type must be "good". '
        "For abnormal images, choose the best allowed defect_type and write one concise sentence explaining the visible evidence."
        + extra
    )


def convert_row(row: dict, dataset_key: str, dataset_name: str, split: str, all_rows: list[dict]) -> dict:
    label = str(row.get("label", "")).lower()
    if label not in {"normal", "abnormal"}:
        label = "abnormal" if int(row.get("label_int", 0) or 0) == 1 else "normal"
    category = str(row.get("category") or ("ksdd2" if dataset_key == "ksdd2_mvtlike" else "unknown"))
    defect_type = normalize_defect_type(row, dataset_key)
    explanation = explanation_for(row, dataset_key, label, defect_type)
    allowed = allowed_types_for(all_rows, dataset_key, category)
    if defect_type and defect_type not in allowed:
        allowed = sorted(set(allowed) | {defect_type})
    target = {"label": label, "defect_type": defect_type, "explanation": explanation}
    ref_images = list(row.get("ref_images") or [])
    return {
        "id": row.get("id"),
        "dataset": dataset_key.replace("_80p", "").replace("_mvtlike", ""),
        "split": split,
        "image": row.get("image"),
        "ref_images": ref_images,
        "category": category,
        "allowed_defect_types": allowed,
        "target": target,
        "prompt": make_prompt(dataset_key, dataset_name, category, len(ref_images), allowed),
        "response": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
        "source": {
            "source_split": row.get("split"),
            "source": row.get("source"),
            "text_source": row.get("text_source"),
            "text_path": row.get("text_path"),
            "text_key": row.get("text_key"),
            "fixed_experiment_split": row.get("split2") or split,
        },
    }


def summarize_rows(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "labels": dict(sorted(Counter(row["target"]["label"] for row in rows).items())),
        "top_defect_types": dict(Counter(row["target"]["defect_type"] for row in rows).most_common(40)),
        "categories": dict(sorted(Counter(row["category"] for row in rows).items())),
    }


def convert_standard_dataset(dataset_key: str, dataset_name: str, manifest_name: str) -> dict:
    train_rows = read_jsonl(SRC_QWEN / dataset_key / "train.jsonl")
    val_rows = read_jsonl(MANIFEST_ROOT / "val" / manifest_name)
    test_rows = read_jsonl(MANIFEST_ROOT / "test" / manifest_name)
    all_rows = train_rows + val_rows + test_rows
    out_dir = OUT_ROOT / dataset_key
    summary = {
        "dataset": dataset_name,
        "train_source": str(SRC_QWEN / dataset_key / "train.jsonl"),
        "val_source": str(MANIFEST_ROOT / "val" / manifest_name),
        "test_source": str(MANIFEST_ROOT / "test" / manifest_name),
        "split_policy": "reuse fixed train/val/test from previous hybrid/typehit experiments",
        "splits": {},
        "categories": {},
    }
    for split, rows in {"train": train_rows, "val": val_rows, "test": test_rows}.items():
        converted = [convert_row(row, dataset_key, dataset_name, split, all_rows) for row in rows]
        write_jsonl(out_dir / f"{split}.jsonl", converted)
        summary["splits"][split] = summarize_rows(converted)
    for category in sorted({row.get("category") for row in all_rows if row.get("category")}):
        summary["categories"][category] = {"allowed_defect_types": allowed_types_for(all_rows, dataset_key, category)}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS) if path.exists() else []


def convert_ksdd2() -> dict:
    train_good = list_images(SRC_ADAPTERS / "ksdd2_mvtlike" / "ksdd2" / "train" / "good")
    ref_images = [str(p) for p in train_good[:3]]
    train_rows = [
        {
            "id": f"train_ksdd2_normal_{p.stem}",
            "dataset": "ksdd2",
            "split": "train",
            "image": str(p),
            "ref_images": [x for x in ref_images if x != str(p)][:3],
            "category": "ksdd2",
            "label": "normal",
            "label_int": 0,
            "source": "ksdd2_mvtlike_train_good",
        }
        for p in train_good
    ]
    val_rows = read_jsonl(MANIFEST_ROOT / "val" / "ksdd2.jsonl")
    test_rows = read_jsonl(MANIFEST_ROOT / "test" / "ksdd2.jsonl")
    all_rows = train_rows + val_rows + test_rows
    out_dir = OUT_ROOT / "ksdd2_mvtlike"
    summary = {
        "dataset": "KSDD2",
        "train_source": str(SRC_ADAPTERS / "ksdd2_mvtlike" / "ksdd2" / "train" / "good"),
        "val_source": str(MANIFEST_ROOT / "val" / "ksdd2.jsonl"),
        "test_source": str(MANIFEST_ROOT / "test" / "ksdd2.jsonl"),
        "note": "defect_type and explanation are intentionally empty for every sample",
        "splits": {},
    }
    for split, rows in {"train": train_rows, "val": val_rows, "test": test_rows}.items():
        converted = [convert_row(row, "ksdd2_mvtlike", "KSDD2", split, all_rows) for row in rows]
        write_jsonl(out_dir / f"{split}.jsonl", converted)
        summary["splits"][split] = summarize_rows(converted)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def validate_schema() -> dict:
    expected = {"label", "defect_type", "explanation"}
    result = {}
    for path in sorted(OUT_ROOT.glob("*/*.jsonl")):
        n = 0
        bad = 0
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            n += 1
            row = json.loads(line)
            try:
                response = json.loads(row["response"])
            except Exception:
                bad += 1
                continue
            if set(row["target"]) != expected or set(response) != expected:
                bad += 1
        result[str(path.relative_to(OUT_ROOT))] = {"n": n, "bad_schema": bad}
    return result


def write_readme_and_csv(summaries: dict) -> None:
    readme = """# Three-field Qwen SFT data

This directory contains simplified Qwen SFT manifests with the fixed train/val/test splits used in the previous experiments.

Each supervised response has exactly three fields:

```json
{"label":"normal|abnormal","defect_type":"...","explanation":"one concise sentence"}
```

Split policy:

- GoodsAD, MVTec AD, VisA, MVTec LOCO: train comes from `data/data_qwen/<dataset>/train.jsonl`; val/test come from `data/splits/typehit_optimization_v5_full_short_loco/manifests/{val,test}`.
- KSDD2: train uses existing normal train images from `data_adapters/ksdd2_mvtlike/ksdd2/train/good`; val/test reuse the same fixed experiment manifests. `defect_type` and `explanation` are empty strings.
"""
    (OUT_ROOT / "README.md").write_text(readme, encoding="utf-8")
    with (OUT_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "split", "n", "normal", "abnormal"])
        writer.writeheader()
        for dataset, summary in summaries.items():
            for split, info in summary["splits"].items():
                labels = info["labels"]
                writer.writerow(
                    {
                        "dataset": dataset,
                        "split": split,
                        "n": info["n"],
                        "normal": labels.get("normal", 0),
                        "abnormal": labels.get("abnormal", 0),
                    }
                )


def main() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = {
        key: convert_standard_dataset(key, cfg["name"], cfg["manifest"])
        for key, cfg in DATASETS.items()
    }
    summaries["ksdd2_mvtlike"] = convert_ksdd2()
    (OUT_ROOT / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme_and_csv(summaries)
    schema = validate_schema()
    print(json.dumps({"summaries": {k: v["splits"] for k, v in summaries.items()}, "schema": schema}, ensure_ascii=False, indent=2))
    print(f"[done] {OUT_ROOT}")


if __name__ == "__main__":
    main()
