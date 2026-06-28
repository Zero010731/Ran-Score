from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from generate_qwen_rationales import DEFAULT_INPUT, validate_rationale


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_source_rows(path: Path) -> dict[str, dict[str, str]]:
    with resolve_project_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit generated Qwen rationale JSONL quality.")
    parser.add_argument("rationale_file", type=Path)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()

    source_rows = load_source_rows(args.input)
    rationale_path = resolve_project_path(args.rationale_file)
    counts = Counter()
    examples: dict[str, list[dict]] = {}

    with rationale_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            row = source_rows[item["sample_id"]]
            try:
                validate_rationale(row, item["rationale"])
                counts["ok"] += 1
            except ValueError as exc:
                reason = str(exc)
                counts[reason] += 1
                examples.setdefault(reason, [])
                if len(examples[reason]) < 3:
                    examples[reason].append(
                        {
                            "line": line_no,
                            "sample_id": item["sample_id"],
                            "rationale": item["rationale"],
                        }
                    )

    print(json.dumps({"counts": counts, "examples": examples}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
