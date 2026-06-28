#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check sample/study/text leakage across RanAudit splits.

The script supports both sample-level classification data and pair-level
LLaMAFactory SFT/DPO data. DPO files may contain multiple rows per sample_id;
that is expected and is reported separately from cross-split leakage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")


def read_jsonl(path: Path, skip_invalid_lines: bool = False) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                if skip_invalid_lines:
                    print(f"[warn] skip invalid JSONL line: {path}:{line_no}: {exc}")
                    continue
                raise ValueError(f"{path}:{line_no} is invalid JSONL: {exc}") from exc
    return rows


def read_csv_or_tsv(path: Path) -> list[dict[str, Any]]:
    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def read_rows(path: Path, skip_invalid_lines: bool = False) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path, skip_invalid_lines=skip_invalid_lines)
    if path.suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return obj
        for key in ("data", "examples", "records"):
            if isinstance(obj, dict) and isinstance(obj.get(key), list):
                return obj[key]
        raise ValueError(f"Unsupported JSON structure: {path}")
    if path.suffix in {".csv", ".tsv"}:
        return read_csv_or_tsv(path)
    raise ValueError(f"Unsupported file type: {path}")


def find_split_file(data_dir: Path, prefix: str, split: str) -> Path | None:
    split_aliases = {
        "train": ("train",),
        "val": ("val", "valid", "validation", "dev"),
        "test": ("test",),
    }[split]
    exts = (".jsonl", ".json", ".csv", ".tsv")
    candidates = []
    for alias in split_aliases:
        if prefix:
            candidates.extend(
                [
                    data_dir / f"{prefix}_{alias}{ext}"
                    for ext in exts
                ]
            )
        candidates.extend(data_dir / f"{alias}{ext}" for ext in exts)
    for path in candidates:
        if path.exists():
            return path
    return None


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[_]{2,}", "_", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def hash_text(text: str) -> str:
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()


def extract_report_from_input(text: str) -> str:
    match = re.search(r"Report:\s*(.*?)(?:\nFinding:|\sFinding:|\s\[SEP\]\s*Finding:)", text, flags=re.S)
    return match.group(1).strip() if match else text.strip()


def get_text(row: dict[str, Any]) -> str:
    for key in ("input", "input_text", "text", "prompt", "source", "query"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    pieces = []
    for key in ("report", "finding_category", "finding_definition", "assigned_label_text"):
        value = row.get(key)
        if value not in (None, ""):
            pieces.append(f"{key}: {value}")
    return "\n".join(pieces)


def record_key(row: dict[str, Any], row_idx: int, split: str) -> dict[str, str]:
    text = get_text(row)
    report = row.get("report") or extract_report_from_input(text)
    return {
        "row_idx": str(row_idx),
        "sample_id": str(row.get("sample_id") or row.get("id") or "").strip(),
        "study_id": str(row.get("study_id") or "").strip(),
        "subject_id": str(row.get("subject_id") or "").strip(),
        "input_hash": hash_text(text) if text else "",
        "report_hash": hash_text(str(report)) if report else "",
        "finding": str(row.get("finding_category") or row.get("finding") or "").strip(),
        "split_field": str(row.get("split") or split).strip(),
    }


def load_split_records(
    data_dir: Path, prefix: str, skip_invalid_lines: bool
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Path]]:
    by_split = {}
    paths = {}
    for split in SPLITS:
        path = find_split_file(data_dir, prefix, split)
        if path is None:
            continue
        paths[split] = path
        by_split[split] = read_rows(path, skip_invalid_lines=skip_invalid_lines)
    if not by_split:
        raise FileNotFoundError(f"No split files found under {data_dir} with prefix={prefix!r}")
    return by_split, paths


def summarize_split(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary = {}
    for split, rows in rows_by_split.items():
        keys = [record_key(row, idx, split) for idx, row in enumerate(rows)]
        summary[split] = {
            "rows": len(rows),
            "unique_sample_id": len({k["sample_id"] for k in keys if k["sample_id"]}),
            "unique_study_id": len({k["study_id"] for k in keys if k["study_id"]}),
            "unique_input_hash": len({k["input_hash"] for k in keys if k["input_hash"]}),
            "unique_report_hash": len({k["report_hash"] for k in keys if k["report_hash"]}),
            "duplicate_sample_id_within_split": sum(
                count - 1
                for count in Counter(k["sample_id"] for k in keys if k["sample_id"]).values()
                if count > 1
            ),
            "split_field_mismatch": sum(1 for k in keys if k["split_field"] and k["split_field"] != split),
        }
    return summary


def collect_values(
    rows_by_split: dict[str, list[dict[str, Any]]], field: str
) -> dict[str, dict[str, list[dict[str, str]]]]:
    values = {}
    for split, rows in rows_by_split.items():
        bucket = defaultdict(list)
        for idx, row in enumerate(rows):
            key = record_key(row, idx, split)
            value = key[field]
            if value:
                bucket[value].append(key)
        values[split] = dict(bucket)
    return values


def cross_split_overlaps(
    rows_by_split: dict[str, list[dict[str, Any]]], field: str, max_examples: int
) -> dict[str, Any]:
    values = collect_values(rows_by_split, field)
    out = {}
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for left, right in pairs:
        left_values = set(values.get(left, {}))
        right_values = set(values.get(right, {}))
        overlap = sorted(left_values & right_values)
        examples = []
        for value in overlap[:max_examples]:
            examples.append(
                {
                    "value": value,
                    left: values[left][value][:3],
                    right: values[right][value][:3],
                }
            )
        out[f"{left}_vs_{right}"] = {
            "overlap_count": len(overlap),
            "examples": examples,
        }
    return out


def check_dataset(
    data_dir: Path,
    prefix: str,
    name: str,
    max_examples: int,
    skip_invalid_lines: bool,
) -> dict[str, Any]:
    rows_by_split, paths = load_split_records(data_dir, prefix, skip_invalid_lines)
    result = {
        "name": name,
        "data_dir": str(data_dir),
        "prefix": prefix,
        "paths": {split: str(path) for split, path in paths.items()},
        "split_summary": summarize_split(rows_by_split),
        "cross_split_overlap": {},
    }
    for field in ("sample_id", "study_id", "input_hash", "report_hash"):
        result["cross_split_overlap"][field] = cross_split_overlaps(rows_by_split, field, max_examples)
    return result


def print_findings(report: dict[str, Any]) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n=== Leakage Finding Summary ===")
    has_leak = False
    for dataset in report["datasets"]:
        print(f"\n[{dataset['name']}]")
        for field, pairs in dataset["cross_split_overlap"].items():
            for pair_name, item in pairs.items():
                count = item["overlap_count"]
                if count:
                    has_leak = True
                    print(f"LEAK? {field} {pair_name}: {count}")
        if not any(
            item["overlap_count"]
            for pairs in dataset["cross_split_overlap"].values()
            for item in pairs.values()
        ):
            print("No cross-split overlaps found for sample_id, study_id, input_hash, or report_hash.")
    print("\nOVERALL:", "POTENTIAL LEAKAGE FOUND" if has_leak else "NO CROSS-SPLIT LEAKAGE DETECTED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert-data-dir", type=Path, default=Path("data/bert_label_audit"))
    parser.add_argument("--dpo-data-dir", type=Path, default=Path("data/llamafactory_qwen_v3_alspc_dpo_v2"))
    parser.add_argument("--sft-prefix", default="ranaudit_sft")
    parser.add_argument("--dpo-prefix", default="ranaudit_dpo")
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_leakage_report.json"))
    parser.add_argument("--skip-invalid-lines", action="store_true")
    args = parser.parse_args()

    datasets = []
    if args.bert_data_dir.exists():
        datasets.append(
            check_dataset(
                args.bert_data_dir,
                "",
                "encoder_sample_level",
                args.max_examples,
                args.skip_invalid_lines,
            )
        )
    else:
        print(f"[skip] missing bert data dir: {args.bert_data_dir}")
    if args.dpo_data_dir.exists():
        datasets.append(
            check_dataset(
                args.dpo_data_dir,
                args.sft_prefix,
                "llamafactory_sft_sample_level",
                args.max_examples,
                args.skip_invalid_lines,
            )
        )
        datasets.append(
            check_dataset(
                args.dpo_data_dir,
                args.dpo_prefix,
                "llamafactory_dpo_pair_level",
                args.max_examples,
                args.skip_invalid_lines,
            )
        )
    else:
        print(f"[skip] missing dpo data dir: {args.dpo_data_dir}")

    report = {"datasets": datasets}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_findings(report)
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
