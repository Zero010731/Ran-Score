import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ALLOWED_CORRECTNESS = {"correct", "incorrect"}
ALLOWED_ERROR_TYPES = {"none", "false_positive", "false_negative"}


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSONL: {exc}") from exc
    return rows


def parse_payload(text):
    if text is None:
        return None
    text = str(text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def expected_correctness(error_type):
    if error_type == "none":
        return "correct"
    if error_type in {"false_positive", "false_negative"}:
        return "incorrect"
    return None


def is_schema_valid(payload):
    if not isinstance(payload, dict):
        return False
    correctness = payload.get("correctness")
    error_type = payload.get("error_type")
    if correctness not in ALLOWED_CORRECTNESS:
        return False
    if error_type not in ALLOWED_ERROR_TYPES:
        return False
    return correctness == expected_correctness(error_type)


def pct(num, den):
    return round(100.0 * num / den, 2) if den else 0.0


def compact_report_text(row, max_chars=500):
    text = row.get("input", "")
    text = text.replace("\n", " ")
    return text[:max_chars]


def load_prediction_records(gold_rows, pred_rows, model_name):
    if len(gold_rows) != len(pred_rows):
        raise ValueError(
            f"{model_name}: gold/pred line count mismatch: {len(gold_rows)} != {len(pred_rows)}"
        )

    records = []
    for idx, (gold_row, pred_row) in enumerate(zip(gold_rows, pred_rows)):
        gold_payload = parse_payload(pred_row.get("label"))
        if gold_payload is None:
            gold_payload = parse_payload(gold_row.get("output"))
        pred_payload = parse_payload(
            pred_row.get("predict")
            or pred_row.get("prediction")
            or pred_row.get("generated_text")
            or pred_row.get("response")
        )

        if gold_payload is None:
            raise ValueError(f"{model_name}: unable to parse gold payload at index {idx}")

        gold_correctness = gold_payload.get("correctness")
        gold_error_type = gold_payload.get("error_type")
        pred_correctness = pred_payload.get("correctness") if isinstance(pred_payload, dict) else None
        pred_error_type = pred_payload.get("error_type") if isinstance(pred_payload, dict) else None

        records.append(
            {
                "idx": idx,
                "sample_id": gold_row.get("sample_id", f"sample_{idx}"),
                "study_id": gold_row.get("study_id", ""),
                "finding_category": gold_row.get("finding_category", ""),
                "assigned_label_text": gold_row.get("assigned_label_text", ""),
                "reference_label_text": gold_row.get("reference_label_text", ""),
                "error_mode": gold_row.get("error_mode", ""),
                "s_neg": gold_row.get("s_neg", ""),
                "s_imp": gold_row.get("s_imp", ""),
                "gold_correctness": gold_correctness,
                "gold_error_type": gold_error_type,
                "pred_correctness": pred_correctness,
                "pred_error_type": pred_error_type,
                "gold_rationale": gold_payload.get("rationale", ""),
                "pred_rationale": pred_payload.get("rationale", "") if isinstance(pred_payload, dict) else "",
                "pred_text": pred_row.get("predict", ""),
                "valid_json": pred_payload is not None,
                "schema_valid": is_schema_valid(pred_payload),
                "correctness_ok": pred_correctness == gold_correctness,
                "error_type_ok": pred_error_type == gold_error_type,
                "joint_ok": pred_correctness == gold_correctness and pred_error_type == gold_error_type,
                "report_snippet": compact_report_text(gold_row),
            }
        )
    return records


def summarize(records):
    counts = Counter()
    by_gold = defaultdict(Counter)
    by_finding = defaultdict(Counter)
    by_assigned = defaultdict(Counter)
    confusion = defaultdict(Counter)
    invalid_error_values = Counter()

    for rec in records:
        gold_err = rec["gold_error_type"]
        pred_err = rec["pred_error_type"]
        finding = rec["finding_category"]
        assigned = rec["assigned_label_text"]

        counts["total"] += 1
        by_gold[gold_err]["total"] += 1
        by_finding[finding]["total"] += 1
        by_assigned[assigned]["total"] += 1
        confusion[gold_err][pred_err] += 1

        if rec["valid_json"]:
            counts["valid_json"] += 1
            by_gold[gold_err]["valid_json"] += 1
        if rec["schema_valid"]:
            counts["schema_valid"] += 1
            by_gold[gold_err]["schema_valid"] += 1
        else:
            if pred_err not in ALLOWED_ERROR_TYPES:
                invalid_error_values[str(pred_err)] += 1

        if rec["correctness_ok"]:
            counts["correctness_ok"] += 1
            by_gold[gold_err]["correctness_ok"] += 1
        if rec["error_type_ok"]:
            counts["error_type_ok"] += 1
            by_gold[gold_err]["error_type_ok"] += 1
        if rec["joint_ok"]:
            counts["joint_ok"] += 1
            by_gold[gold_err]["joint_ok"] += 1
            by_finding[finding]["joint_ok"] += 1
            by_assigned[assigned]["joint_ok"] += 1
        else:
            by_finding[finding]["wrong"] += 1
            by_assigned[assigned]["wrong"] += 1

        if gold_err == "none" and rec["pred_correctness"] == "incorrect":
            counts["false_alarm"] += 1
        if gold_err != "none" and pred_err == "none":
            counts["missed_error"] += 1

    result = {
        "total": counts["total"],
        "valid_json_rate": pct(counts["valid_json"], counts["total"]),
        "schema_valid_rate": pct(counts["schema_valid"], counts["total"]),
        "correctness_accuracy": pct(counts["correctness_ok"], counts["total"]),
        "error_type_accuracy": pct(counts["error_type_ok"], counts["total"]),
        "joint_accuracy": pct(counts["joint_ok"], counts["total"]),
        "false_alarm_rate_on_none": pct(counts["false_alarm"], by_gold["none"]["total"]),
        "missed_error_rate_on_error_samples": pct(
            counts["missed_error"],
            by_gold["false_positive"]["total"] + by_gold["false_negative"]["total"],
        ),
        "invalid_error_type_values": dict(invalid_error_values),
        "by_gold_error_type": {},
        "confusion_error_type": {str(k): dict(v) for k, v in confusion.items()},
        "worst_findings_by_error_count": [],
        "by_assigned_label": {},
    }

    for key in ["false_positive", "false_negative", "none"]:
        c = by_gold[key]
        result["by_gold_error_type"][key] = {
            "n": c["total"],
            "valid_json_rate": pct(c["valid_json"], c["total"]),
            "schema_valid_rate": pct(c["schema_valid"], c["total"]),
            "joint_accuracy": pct(c["joint_ok"], c["total"]),
        }

    finding_rows = []
    for finding, c in by_finding.items():
        if c["total"]:
            finding_rows.append(
                {
                    "finding": finding,
                    "n": c["total"],
                    "wrong": c["wrong"],
                    "joint_accuracy": pct(c["joint_ok"], c["total"]),
                }
            )
    result["worst_findings_by_error_count"] = sorted(
        finding_rows, key=lambda x: (-x["wrong"], x["finding"])
    )[:20]

    for assigned, c in sorted(by_assigned.items()):
        result["by_assigned_label"][assigned] = {
            "n": c["total"],
            "wrong": c["wrong"],
            "joint_accuracy": pct(c["joint_ok"], c["total"]),
        }

    return result


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def md_table(headers, rows):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def write_markdown_report(path, model_summaries, comparisons):
    lines = ["# Prediction Error Analysis", ""]

    overview_rows = []
    for name, summary in model_summaries.items():
        overview_rows.append(
            [
                name,
                summary["total"],
                f'{summary["valid_json_rate"]:.2f}',
                f'{summary["schema_valid_rate"]:.2f}',
                f'{summary["joint_accuracy"]:.2f}',
                f'{summary["false_alarm_rate_on_none"]:.2f}',
                f'{summary["missed_error_rate_on_error_samples"]:.2f}',
            ]
        )
    lines.append("## Overall")
    lines.append("")
    lines.append(
        md_table(
            [
                "model",
                "n",
                "json_valid",
                "schema_valid",
                "joint_acc",
                "false_alarm_none",
                "missed_error",
            ],
            overview_rows,
        )
    )
    lines.append("")

    for name, summary in model_summaries.items():
        lines.append(f"## {name}")
        lines.append("")
        rows = []
        for key, stats in summary["by_gold_error_type"].items():
            rows.append(
                [
                    key,
                    stats["n"],
                    f'{stats["schema_valid_rate"]:.2f}',
                    f'{stats["joint_accuracy"]:.2f}',
                ]
            )
        lines.append(md_table(["gold_error_type", "n", "schema_valid", "joint_acc"], rows))
        lines.append("")
        if summary["invalid_error_type_values"]:
            lines.append(f"Invalid error_type values: `{summary['invalid_error_type_values']}`")
            lines.append("")
        lines.append("Worst findings by error count:")
        lines.append("")
        finding_rows = [
            [r["finding"], r["n"], r["wrong"], f'{r["joint_accuracy"]:.2f}']
            for r in summary["worst_findings_by_error_count"][:10]
        ]
        lines.append(md_table(["finding", "n", "wrong", "joint_acc"], finding_rows))
        lines.append("")

    if comparisons:
        lines.append("## Comparisons")
        lines.append("")
        for comp in comparisons:
            lines.append(f"### {comp['base']} -> {comp['model']}")
            lines.append("")
            lines.append(
                md_table(
                    ["corrected", "regressed", "both_wrong", "both_right"],
                    [[comp["corrected"], comp["regressed"], comp["both_wrong"], comp["both_right"]]],
                )
            )
            lines.append("")

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_records(base_name, base_records, model_name, model_records):
    if len(base_records) != len(model_records):
        raise ValueError(f"Cannot compare {base_name} and {model_name}: length mismatch.")

    corrected = []
    regressed = []
    both_wrong = []
    both_right = []
    for base, model in zip(base_records, model_records):
        if not base["joint_ok"] and model["joint_ok"]:
            corrected.append(model)
        elif base["joint_ok"] and not model["joint_ok"]:
            regressed.append(model)
        elif not base["joint_ok"] and not model["joint_ok"]:
            both_wrong.append(model)
        else:
            both_right.append(model)

    return {
        "base": base_name,
        "model": model_name,
        "corrected": len(corrected),
        "regressed": len(regressed),
        "both_wrong": len(both_wrong),
        "both_right": len(both_right),
        "corrected_rows": corrected,
        "regressed_rows": regressed,
        "both_wrong_rows": both_wrong,
    }


def parse_pred_arg(value):
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    return name, Path(path)


def main():
    parser = argparse.ArgumentParser(description="Analyze RanAudit prediction errors.")
    parser.add_argument("--gold", required=True, help="Gold SFT JSONL file.")
    parser.add_argument(
        "--pred",
        action="append",
        required=True,
        help="Prediction JSONL. Use name=path to set model name. Can be repeated.",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for reports and error files.")
    parser.add_argument(
        "--base",
        default=None,
        help="Optional base model name for correction/regression comparisons. Defaults to first pred.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_rows = read_jsonl(args.gold)
    model_records = {}
    model_summaries = {}

    for pred_arg in args.pred:
        name, path = parse_pred_arg(pred_arg)
        pred_rows = read_jsonl(path)
        records = load_prediction_records(gold_rows, pred_rows, name)
        summary = summarize(records)
        model_records[name] = records
        model_summaries[name] = summary

        model_dir = out_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        wrong_rows = [row for row in records if not row["joint_ok"]]
        write_jsonl(model_dir / "errors.jsonl", wrong_rows)
        write_csv(model_dir / "errors.csv", wrong_rows)
        (model_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    comparisons = []
    if len(model_records) >= 2:
        base_name = args.base or next(iter(model_records))
        if base_name not in model_records:
            raise ValueError(f"Base model {base_name!r} not found in predictions.")
        for name, records in model_records.items():
            if name == base_name:
                continue
            comp = compare_records(base_name, model_records[base_name], name, records)
            comparisons.append(comp)
            comp_dir = out_dir / f"{base_name}_vs_{name}"
            comp_dir.mkdir(parents=True, exist_ok=True)
            write_csv(comp_dir / "corrected.csv", comp["corrected_rows"])
            write_csv(comp_dir / "regressed.csv", comp["regressed_rows"])
            write_csv(comp_dir / "both_wrong.csv", comp["both_wrong_rows"])
            compact = {k: v for k, v in comp.items() if not k.endswith("_rows")}
            (comp_dir / "summary.json").write_text(
                json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

    write_markdown_report(out_dir / "error_analysis_report.md", model_summaries, comparisons)
    print(json.dumps(model_summaries, ensure_ascii=False, indent=2))
    print(f"Saved report to {out_dir / 'error_analysis_report.md'}")


if __name__ == "__main__":
    main()
