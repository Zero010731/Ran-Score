#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VALID_CORRECTNESS = {"correct", "incorrect"}
VALID_ERROR_TYPE = {"none", "false_positive", "false_negative"}


def parse_json_field(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None

    text = value.strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
    return None


def pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def compute_metrics(pred_path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in pred_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    total = 0
    valid_json = 0
    schema_valid = 0
    correctness_ok = 0
    error_type_ok = 0
    joint_ok = 0
    none_total = 0
    false_alarm_none = 0
    error_total = 0
    missed_error = 0
    invalid_examples: list[dict[str, Any]] = []
    by_gold: dict[str, dict[str, int]] = {}
    confusion: dict[str, dict[str, int]] = {}

    for idx, row in enumerate(rows):
        gold = parse_json_field(row.get("label"))
        pred = parse_json_field(
            row.get("predict")
            or row.get("prediction")
            or row.get("response")
            or row.get("generated_text")
        )
        if gold is None:
            raise ValueError(f"{pred_path}:{idx + 1} is missing a parseable label field.")

        total += 1
        gold_correctness = gold.get("correctness")
        gold_error_type = gold.get("error_type")
        pred_correctness = pred.get("correctness") if pred else None
        pred_error_type = pred.get("error_type") if pred else None

        bucket = by_gold.setdefault(str(gold_error_type), {"n": 0, "joint_ok": 0})
        bucket["n"] += 1

        if pred is None:
            if len(invalid_examples) < 5:
                invalid_examples.append(
                    {"idx": idx, "prediction": row.get("predict") or row.get("prediction")}
                )
            continue

        valid_json += 1
        schema_good = (
            pred_correctness in VALID_CORRECTNESS
            and pred_error_type in VALID_ERROR_TYPE
            and not (pred_correctness == "correct" and pred_error_type != "none")
            and not (pred_correctness == "incorrect" and pred_error_type == "none")
        )
        if schema_good:
            schema_valid += 1

        if pred_correctness == gold_correctness:
            correctness_ok += 1
        if pred_error_type == gold_error_type:
            error_type_ok += 1
        if pred_correctness == gold_correctness and pred_error_type == gold_error_type:
            joint_ok += 1
            bucket["joint_ok"] += 1

        if gold_error_type == "none":
            none_total += 1
            if pred_error_type != "none":
                false_alarm_none += 1
        elif gold_error_type in {"false_positive", "false_negative"}:
            error_total += 1
            if pred_error_type == "none":
                missed_error += 1

        pred_key = str(pred_error_type)
        confusion.setdefault(str(gold_error_type), {})
        confusion[str(gold_error_type)][pred_key] = (
            confusion[str(gold_error_type)].get(pred_key, 0) + 1
        )

    return {
        "total": total,
        "valid_json_rate": pct(valid_json, total),
        "schema_valid_rate": pct(schema_valid, total),
        "correctness_accuracy": pct(correctness_ok, total),
        "error_type_accuracy": pct(error_type_ok, total),
        "joint_accuracy": pct(joint_ok, total),
        "false_alarm_rate_on_none": pct(false_alarm_none, none_total),
        "missed_error_rate_on_error_samples": pct(missed_error, error_total),
        "by_gold_error_type": {
            key: {"n": value["n"], "joint_accuracy": pct(value["joint_ok"], value["n"])}
            for key, value in sorted(by_gold.items())
        },
        "confusion_error_type": confusion,
        "invalid_examples": invalid_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RanAudit generated predictions.")
    parser.add_argument(
        "--pred-path",
        type=Path,
        required=True,
        help="Path to generated_predictions.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults to task_metrics.json beside predictions.",
    )
    args = parser.parse_args()

    metrics = compute_metrics(args.pred_path)
    output_path = args.output or args.pred_path.parent / "task_metrics.json"
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
