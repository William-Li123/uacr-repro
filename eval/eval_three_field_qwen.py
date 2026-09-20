from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__import__("os").environ["UACR_ROOT"])
DATA_ROOT = ROOT / "data" / "three_field_qwen"
DEFAULT_MODEL = ROOT / "model_qwen_base" / "Qwen3.5-9B"
REQUIRED_KEYS = {"label", "defect_type", "explanation"}

STOPWORDS = {
    "the", "and", "or", "a", "an", "is", "are", "it", "its", "of", "to", "in", "on", "at", "as",
    "with", "from", "by", "for", "this", "that", "query", "image", "normal", "abnormal", "defect",
    "defects", "reference", "references", "visible", "shows", "show", "compared", "product", "object",
    "area", "part", "parts", "surface", "appearance", "there", "has", "have", "having",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"normal", "good", "0"}:
        return "normal"
    if text in {"abnormal", "defect", "defective", "anomaly", "1"}:
        return "abnormal"
    return "unknown"


def norm_type(value: Any) -> str:
    if isinstance(value, list):
        return "__".join(norm_type(item) for item in value if norm_type(item))
    text = str(value or "").strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if text in {"none", "null", "na", "n_a", "unknown"}:
        return ""
    return text


def split_defect_types(value: Any) -> list[str]:
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(split_defect_types(item))
        return out
    raw = str(value or "").strip()
    if not raw:
        return []
    parts: list[str] = []
    for chunk in re.split(r"__|[,;/|]+", raw):
        token = norm_type(chunk)
        if token:
            parts.append(token)
    return parts


def strip_code_fence(text: str) -> str:
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def extract_json(text: str) -> tuple[dict[str, Any], str]:
    raw = strip_code_fence(text)
    start = raw.find("{")
    if start < 0:
        return {}, raw
    depth = 0
    in_str = False
    esc = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = raw[start : idx + 1]
                try:
                    obj = json.loads(candidate)
                except Exception:
                    return {}, raw
                return (obj if isinstance(obj, dict) else {}), raw
    return {}, raw


def parse_prediction(text: str) -> dict[str, Any]:
    obj, cleaned = extract_json(text)
    low = cleaned.lower()
    label = norm_label(obj.get("label"))
    if label == "unknown":
        if re.search(r"\babnormal\b|\bdefect\b|\banomaly\b", low):
            label = "abnormal"
        elif re.search(r"\bnormal\b|\bgood\b", low):
            label = "normal"
    defect_type = norm_type(obj.get("defect_type"))
    explanation = str(obj.get("explanation") or "").strip()
    return {
        "valid_json": int(bool(obj)),
        "exact_schema": int(set(obj.keys()) == REQUIRED_KEYS),
        "pred_label": label,
        "pred_defect_type": defect_type,
        "pred_explanation": explanation,
    }


def tokenize_explanation(text: str) -> set[str]:
    text = str(text or "").lower().replace("_", " ")
    tokens = set()
    for token in re.findall(r"[a-z][a-z0-9]+", text):
        if len(token) >= 3 and token not in STOPWORDS:
            tokens.add(token)
    return tokens


def explanation_keyword_hit(target_explanation: str, target_defect_type: str, pred_explanation: str) -> tuple[int, int, str]:
    target_terms = tokenize_explanation(target_explanation) | tokenize_explanation(target_defect_type)
    pred_terms = tokenize_explanation(pred_explanation)
    if not target_terms:
        return 0, 0, ""
    overlap = sorted(target_terms & pred_terms)
    required = 1 if len(target_terms) < 6 else 2
    return int(len(overlap) >= required), len(overlap), " ".join(sorted(target_terms))


def build_prompt(row: dict[str, Any], prompt_source: str) -> str:
    if prompt_source == "row":
        prompt = str(row.get("prompt") or "").strip()
        if prompt:
            return prompt
    allowed = row.get("allowed_defect_types") or []
    allowed_text = ", ".join(str(x) for x in allowed) if allowed else "none"
    dataset = row.get("dataset", "unknown")
    category = row.get("category", "unknown")
    if dataset == "ksdd2":
        return (
            "You are a careful industrial visual anomaly inspector. Dataset: KSDD2. Category: steel surface. "
            "The first image(s), if present, are normal reference examples; the last image is the query image. "
            'Return only one compact JSON object with exactly three keys: "label", "defect_type", and "explanation". '
            'The label must be exactly "normal" or "abnormal". For KSDD2 here, "defect_type" and "explanation" '
            "must be empty strings because no reliable fine-grained textual labels are available."
        )
    loco_extra = ""
    if dataset == "mvtec_loco":
        loco_extra = (
            " MVTec LOCO uses coarse defect_type labels: logical_anomaly means wrong count, missing or extra object, "
            "wrong object, or wrong arrangement; structural_anomaly means local damage, contamination, deformation, "
            "or surface/appearance change."
        )
    return (
        f"You are a careful industrial visual anomaly inspector. Dataset: {dataset}. Category: {category}. "
        "The first image(s) are normal reference examples from the same category; the last image is the query image. "
        "Compare the query with the references and decide whether it is normal or abnormal. "
        f"Allowed defect_type values are: {allowed_text}. "
        'Return only one compact JSON object with exactly three keys: "label", "defect_type", and "explanation". '
        'The label must be exactly "normal" or "abnormal". For normal images, defect_type must be "good". '
        "For abnormal images, choose the best allowed defect_type and write one concise sentence explaining the visible evidence."
        + loco_extra
    )


def make_message(row: dict[str, Any], prompt_source: str) -> list[dict[str, Any]]:
    content = [{"type": "image", "image": str(path)} for path in row.get("ref_images", [])]
    content.append({"type": "image", "image": str(row["image"])})
    content.append({"type": "text", "text": build_prompt(row, prompt_source)})
    return [{"role": "user", "content": content}]


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {row["id"] for row in csv.DictReader(f)}


def allowed_type_ok(row: dict[str, Any], pred_type: str) -> int:
    target = row.get("target") or {}
    target_type = norm_type(target.get("defect_type"))
    allowed = [norm_type(x) for x in (row.get("allowed_defect_types") or [])]
    dataset = row.get("dataset")
    if dataset == "ksdd2" or target_type == "":
        return int(pred_type == "")
    if pred_type == "":
        return 0
    return int(pred_type in allowed)


def row_metrics(row: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    target = row.get("target") or {}
    target_label = norm_label(target.get("label"))
    target_type = norm_type(target.get("defect_type"))
    target_expl = str(target.get("explanation") or "")
    pred_label = pred["pred_label"]
    pred_type = pred["pred_defect_type"]
    target_parts = set(split_defect_types(target_type))
    pred_parts = set(split_defect_types(pred_type))
    type_hit = int(bool(target_parts and pred_parts and target_parts & pred_parts))
    type_exact = int(target_type == pred_type)
    expl_hit, overlap_count, target_terms = explanation_keyword_hit(target_expl, target_type, pred["pred_explanation"])
    return {
        "target_label": target_label,
        "target_defect_type": target_type,
        "target_explanation": target_expl,
        "label_correct": int(target_label == pred_label),
        "defect_type_exact": type_exact,
        "defect_type_hit": type_hit,
        "allowed_type_ok": allowed_type_ok(row, pred_type),
        "explanation_keyword_hit": expl_hit,
        "explanation_overlap_count": overlap_count,
        "explanation_target_terms": target_terms,
        "is_true_abnormal": int(target_label == "abnormal"),
        "has_type_target": int(bool(target_type)),
        "has_explanation_target": int(bool(target_expl)),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    def mean(field: str, subset: list[dict[str, Any]] | None = None) -> float:
        values = subset if subset is not None else rows
        if not values:
            return 0.0
        return sum(float(r[field]) for r in values) / len(values)

    normal = [r for r in rows if r["target_label"] == "normal"]
    abnormal = [r for r in rows if r["target_label"] == "abnormal"]
    abnormal_type = [r for r in abnormal if int(r["has_type_target"]) == 1]
    abnormal_expl = [r for r in abnormal if int(r["has_explanation_target"]) == 1]
    normal_recall = mean("label_correct", normal)
    abnormal_recall = mean("label_correct", abnormal)
    return {
        "n": len(rows),
        "valid_json_rate": mean("valid_json"),
        "exact_schema_rate": mean("exact_schema"),
        "label_accuracy": mean("label_correct"),
        "normal_recall": normal_recall,
        "abnormal_recall": abnormal_recall,
        "balanced_accuracy": (normal_recall + abnormal_recall) / 2 if normal and abnormal else mean("label_correct"),
        "allowed_type_rate": mean("allowed_type_ok"),
        "defect_type_exact_abnormal": mean("defect_type_exact", abnormal_type),
        "defect_type_hit_abnormal": mean("defect_type_hit", abnormal_type),
        "explanation_keyword_hit_abnormal": mean("explanation_keyword_hit", abnormal_expl),
        "elapsed_sec_sum": sum(float(r["elapsed_sec"]) for r in rows),
        "sec_per_image": sum(float(r["elapsed_sec"]) for r in rows) / len(rows),
        "denom_type_abnormal": len(abnormal_type),
        "denom_explanation_abnormal": len(abnormal_expl),
    }


def write_summary(csv_path: Path, summary_path: Path, wall_elapsed: float | None = None) -> None:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    overall = aggregate(rows)
    if wall_elapsed is not None and overall:
        overall["wall_elapsed_sec"] = float(wall_elapsed)
        overall["wall_sec_per_image"] = float(wall_elapsed / max(len(rows), 1))

    by_category_map: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_label_map: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category_map[(row["dataset"], row["category"])].append(row)
        by_label_map[(row["dataset"], row["target_label"])].append(row)

    by_category = [
        {"dataset": key[0], "category": key[1], **aggregate(vals)}
        for key, vals in sorted(by_category_map.items())
    ]
    by_label = [
        {"dataset": key[0], "label": key[1], **aggregate(vals)}
        for key, vals in sorted(by_label_map.items())
    ]
    summary = {"overall": overall, "by_category": by_category, "by_label": by_label}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_table(path: Path, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        keys = list(records[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)

    write_table(summary_path.with_suffix(".by_category.csv"), by_category)
    write_table(summary_path.with_suffix(".by_label.csv"), by_label)


def resolve_manifest(args: argparse.Namespace) -> Path:
    if args.manifest:
        return Path(args.manifest)
    if not args.dataset:
        raise SystemExit("Either --manifest or --dataset must be provided.")
    return Path(args.data_root) / args.dataset / f"{args.split}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen/Qwen-LoRA with the three-field anomaly JSON protocol.")
    parser.add_argument("--manifest", default="", help="Path to train/val/test JSONL. Overrides --dataset/--split.")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--dataset", default="", help="Dataset folder under data-root, e.g. mvtec_ad_80p.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--adapter", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--max-pixels", type=int, default=147456)
    parser.add_argument("--prompt-source", choices=["row", "fallback"], default="row")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    manifest = resolve_manifest(args)
    rows = read_jsonl(manifest)
    if args.limit:
        rows = rows[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(
        json.dumps({**vars(args), "manifest": str(manifest)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    csv_path = out_dir / "predictions.csv"
    summary_path = out_dir / "summary.json"
    fieldnames = [
        "id", "dataset", "split", "category", "image",
        "target_label", "target_defect_type", "target_explanation",
        "pred_label", "pred_defect_type", "pred_explanation",
        "valid_json", "exact_schema", "allowed_type_ok", "label_correct",
        "defect_type_exact", "defect_type_hit", "explanation_keyword_hit",
        "explanation_overlap_count", "explanation_target_terms",
        "is_true_abnormal", "has_type_target", "has_explanation_target",
        "raw_answer", "elapsed_sec",
    ]
    if args.no_resume or not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    done = set() if args.no_resume else existing_ids(csv_path)
    samples = [row for row in rows if str(row.get("id")) not in done]
    print(f"manifest={manifest} total={len(rows)} remaining={len(samples)} out={out_dir}", flush=True)
    if not samples:
        write_summary(csv_path, summary_path)
        return

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=True,
        min_pixels=224 * 224,
        max_pixels=args.max_pixels,
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    device = next(model.parameters()).device

    wall_start = time.perf_counter()
    for start in range(0, len(samples), args.batch_size):
        batch_rows = samples[start : start + args.batch_size]
        messages = [make_message(row, args.prompt_source) for row in batch_rows]
        texts = []
        for message in messages:
            try:
                text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            texts.append(text)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
        infer_start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        elapsed = time.perf_counter() - infer_start
        new_tokens = [out[len(inp) :] for inp, out in zip(inputs.input_ids, output)]
        answers = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        per_image = elapsed / max(len(batch_rows), 1)

        output_rows = []
        for row, answer in zip(batch_rows, answers):
            pred = parse_prediction(answer)
            metrics = row_metrics(row, pred)
            output_rows.append(
                {
                    "id": row.get("id"),
                    "dataset": row.get("dataset"),
                    "split": row.get("split"),
                    "category": row.get("category"),
                    "image": row.get("image"),
                    "pred_label": pred["pred_label"],
                    "pred_defect_type": pred["pred_defect_type"],
                    "pred_explanation": pred["pred_explanation"],
                    "valid_json": pred["valid_json"],
                    "exact_schema": pred["exact_schema"],
                    "raw_answer": answer.strip(),
                    "elapsed_sec": f"{per_image:.6f}",
                    **metrics,
                }
            )
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(output_rows)
        print(f"processed={min(start + len(batch_rows), len(samples))}/{len(samples)}", flush=True)

    write_summary(csv_path, summary_path, wall_elapsed=time.perf_counter() - wall_start)
    print(f"[done] predictions={csv_path} summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
