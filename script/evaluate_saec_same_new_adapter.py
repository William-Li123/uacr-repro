from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__import__("os").environ["UACR_ROOT"])
SCRIPT_ROOT = ROOT / "route_compare_test" / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "script"))

import compare_random50_saec_ours_20260630 as compare  # noqa: E402
import tune_per_dataset_q_20260630 as route  # noqa: E402
from eval_three_field_qwen import aggregate, norm_type, parse_prediction, row_metrics  # noqa: E402
from evaluate_pixel_variant_routes import add_type_p  # noqa: E402


MODELS = ["qwen35_9b", "qwen25vl_7b"]
DATASETS = ["goodsad_80p", "mvtec_ad_80p", "mvtec_loco_80p", "visa_80p", "ksdd2_mvtlike"]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    prediction_root = Path(__import__("os").environ["UACR_PREDICTIONS"])
    result_root = Path(__import__("os").environ["UACR_SAEC_OUTPUT"])
    summaries = []
    for model in MODELS:
        for dataset in DATASETS:
            manifest = route.read_jsonl(route.DATA_ROOT / dataset / "test.jsonl")
            predictions = {
                str(row["id"]): row
                for row in read_csv(prediction_root / model / dataset / "test" / "predictions.csv")
            }
            saec_routes = {
                str(row["id"]): row
                for row in read_csv(ROOT / "route_compare_test" / "routes_cached" / "saec_mod" / f"{dataset}.csv")
            }
            final_rows = []
            for manifest_row in manifest:
                sample_id = str(manifest_row["id"])
                route_row = saec_routes[sample_id]
                called = int(float(route_row.get("qwen_called", 0) or 0))
                if called:
                    qwen_row = predictions[sample_id]
                    prediction = compare.qwen_prediction_from_csv(qwen_row, parse_prediction, norm_type)
                    elapsed = float(qwen_row.get("elapsed_sec") or 0.0)
                    raw_answer = qwen_row.get("raw_answer", "")
                else:
                    prediction = compare.deterministic_normal_prediction(dataset)
                    elapsed = 0.0
                    raw_answer = json.dumps(prediction, ensure_ascii=False)
                final_rows.append(
                    compare.output_row(
                        manifest_row,
                        prediction,
                        "saec_same_new_adapter",
                        model,
                        dataset,
                        called,
                        route_row.get("route_source", ""),
                        elapsed,
                        raw_answer,
                        row_metrics,
                    )
                )
            summary = compare.summarize(model, "saec_same_new_adapter", dataset, final_rows, aggregate)
            summaries.append(add_type_p(summary, final_rows))
            write_csv(result_root / model / f"{dataset}.predictions.csv", final_rows)
    write_csv(result_root / "test_summary.csv", summaries)
    print(json.dumps({"summary": str(result_root / "test_summary.csv")}, indent=2))


if __name__ == "__main__":
    main()
