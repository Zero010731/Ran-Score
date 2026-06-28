from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SFT = {
    "instruction",
    "input",
    "output",
    "sample_id",
    "split",
    "gold_correctness",
    "gold_error_type",
}

REQUIRED_DPO = {
    "instruction",
    "input",
    "chosen",
    "rejected",
    "sample_id",
    "split",
    "gold_correctness",
    "gold_error_type",
    "error_mode",
    "s_neg",
    "s_imp",
    "rejected_type",
}

def load_jsonl(path: Path) -> list[dict]:
    path = resolve_project_path(path)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
    return rows


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_payload(value: str, path: Path, line_no: int, field: str) -> dict:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line_no} field {field} is not JSON: {exc}") from exc

    required = {"correctness", "error_type"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{path}:{line_no} field {field} missing keys: {sorted(missing)}")
    if "rationale" in payload and not str(payload["rationale"]).strip():
        raise ValueError(f"{path}:{line_no} field {field} has empty rationale")
    return payload


def validate_file(
    path: Path,
    required: set[str],
    payload_fields: tuple[str, ...],
    require_rationale: bool,
) -> dict:
    rows = load_jsonl(path)
    split_counts = Counter()
    modes = Counter()
    rejected_types = Counter()

    for idx, row in enumerate(rows, start=1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"{path}:{idx} missing keys: {sorted(missing)}")

        split_counts[row["split"]] += 1
        if "error_mode" in row:
            modes[row["error_mode"]] += 1
        if "rejected_type" in row:
            rejected_types[row["rejected_type"]] += 1

        for field in payload_fields:
            payload = parse_payload(row[field], path, idx, field)
            if payload["correctness"] not in {"correct", "incorrect"}:
                raise ValueError(f"{path}:{idx} invalid correctness in {field}: {payload['correctness']}")
            if payload["error_type"] not in {"none", "false_positive", "false_negative"}:
                raise ValueError(f"{path}:{idx} invalid error_type in {field}: {payload['error_type']}")
            if require_rationale and "rationale" not in payload:
                raise ValueError(f"{path}:{idx} field {field} missing rationale")

            if field in {"output", "chosen"}:
                if payload["correctness"] != row["gold_correctness"]:
                    raise ValueError(
                        f"{path}:{idx} field {field} changed correctness: "
                        f"{payload['correctness']} != {row['gold_correctness']}"
                    )
                if payload["error_type"] != row["gold_error_type"]:
                    raise ValueError(
                        f"{path}:{idx} field {field} changed error_type: "
                        f"{payload['error_type']} != {row['gold_error_type']}"
                    )

        if "chosen" in row and row["chosen"] == row["rejected"]:
            raise ValueError(f"{path}:{idx} chosen and rejected are identical")

    return {
        "path": str(path),
        "rows": len(rows),
        "split_counts": dict(split_counts),
        "error_modes": dict(modes),
        "rejected_types": dict(rejected_types),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated RanAudit LlamaFactory JSONL files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/llamafactory"))
    parser.add_argument(
        "--require-rationale",
        action="store_true",
        help="Require output/chosen/rejected payloads to contain rationales. Kept for legacy full-rationale DPO data.",
    )
    parser.add_argument(
        "--require-sft-rationale",
        action="store_true",
        help="Require SFT output payloads to contain Qwen-generated rationales.",
    )
    parser.add_argument(
        "--require-dpo-rationale",
        action="store_true",
        help="Require DPO chosen/rejected payloads to contain rationales.",
    )
    args = parser.parse_args()

    reports = []
    for split in ("train", "val", "test"):
        reports.append(
            validate_file(
                args.data_dir / f"ranaudit_sft_{split}.jsonl",
                REQUIRED_SFT,
                ("output",),
                args.require_rationale or args.require_sft_rationale,
            )
        )
        reports.append(
            validate_file(
                args.data_dir / f"ranaudit_dpo_{split}.jsonl",
                REQUIRED_DPO,
                ("chosen", "rejected"),
                args.require_rationale or args.require_dpo_rationale,
            )
        )

    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
