from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


DEFAULT_INPUT = Path(
    "data/processed/RanAudit_binary_label_consistency_dataset_balanced_8400_study_split.csv"
)
DEFAULT_OUTPUT_DIR = Path("data/llamafactory")
DEFAULT_RATIONALE_CACHE = Path("data/llamafactory/qwen_rationales.jsonl")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INSTRUCTION = (
    "You are a radiology label auditor. Given a radiology report, a candidate "
    "finding, its definition, and an assigned label, determine whether the "
    "assigned label is consistent with the report. Return JSON only."
)

EXPLICIT_NEGATION_CUES = (
    "no evidence of",
    "negative for",
    "not seen",
    "without",
    "no",
    "absent",
    "free of",
)

WEAK_ABSENCE_CUES = (
    "no acute",
    "clear",
    "normal",
    "unremarkable",
)

NEGATION_CUES = EXPLICIT_NEGATION_CUES + WEAK_ABSENCE_CUES

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "between",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "or",
    "other",
    "the",
    "to",
    "with",
    "within",
}

ERROR_TYPE = {
    "none": "none",
    "negative_to_positive": "false_positive",
    "positive_to_negative": "false_negative",
}

ERROR_MODE = {
    "none": "None",
    "negative_to_positive": "FP",
    "positive_to_negative": "FN",
}


@dataclass(frozen=True)
class RiskSignals:
    s_neg: float
    s_imp: float
    neg_evidence: str
    imp_evidence: str


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower())


def split_sentences(report: str) -> list[str]:
    normalized = report.replace("\n", " ")
    normalized = normalized.replace(":.", ": ")
    normalized = re.sub(r"\.(?=[A-Z_])", ". ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    pieces = re.split(r"(?<=[.!?])\s+", normalized)
    sentences = []
    for piece in pieces:
        sentence = normalize_sentence(piece)
        if sentence:
            sentences.append(sentence)
    return sentences or [normalized]


def normalize_sentence(sentence: str) -> str:
    sentence = sentence.strip(" .")
    header_pattern = (
        r"^(FINAL REPORT|EXAMINATION|INDICATION|COMPARISON|FINDINGS|IMPRESSION|"
        r"HISTORY|TECHNIQUE|CHEST RADIOGRAPH|PORTABLE CHEST|PA AND LATERAL CHEST)"
        r"[:.\s-]*"
    )
    previous = None
    while previous != sentence:
        previous = sentence
        sentence = re.sub(header_pattern, "", sentence, flags=re.IGNORECASE).strip(" .")

    tokens = tokenize(sentence)
    if len(tokens) < 2:
        return ""
    if sentence.upper() in {"FINAL REPORT", "EXAMINATION", "FINDINGS", "IMPRESSION"}:
        return ""
    return sentence


def anchor_terms(finding: str, definition: str) -> set[str]:
    terms = set()
    for token in tokenize(f"{finding} {definition}"):
        token = token.strip("-")
        if len(token) >= 4 and token not in STOPWORDS:
            terms.add(token)
    return terms


def cue_positions(tokens: list[str], cue: str) -> list[int]:
    cue_tokens = cue.split()
    if len(cue_tokens) == 1:
        return [idx for idx, token in enumerate(tokens) if token == cue_tokens[0]]

    positions = []
    width = len(cue_tokens)
    for idx in range(0, max(0, len(tokens) - width + 1)):
        if tokens[idx : idx + width] == cue_tokens:
            positions.append(idx)
    return positions


def negation_score(report: str, anchors: set[str]) -> tuple[float, str]:
    best_sentence = ""
    raw_score = 0.0
    for sentence in split_sentences(report):
        tokens = tokenize(sentence)
        anchor_positions = [
            idx for idx, token in enumerate(tokens) if token in anchors or token.rstrip("s") in anchors
        ]
        sentence_score = 0.0
        for cue in NEGATION_CUES:
            for cue_pos in cue_positions(tokens, cue):
                if anchor_positions:
                    nearest = min(abs(cue_pos - anchor_pos) for anchor_pos in anchor_positions)
                    sentence_score += 1.0 / (nearest + 1.0)
                else:
                    sentence_score += 0.15

        if sentence_score > raw_score:
            raw_score = sentence_score
            best_sentence = sentence

    return 1.0 - math.exp(-raw_score), best_sentence


def implicit_score(report: str, anchors: set[str]) -> tuple[float, str]:
    best_sentence = ""
    best_score = 0.0
    for sentence in split_sentences(report):
        sent_terms = set(tokenize(sentence))
        if not sent_terms or not anchors:
            continue
        overlap = len(sent_terms & anchors)
        score = overlap / math.sqrt(len(sent_terms) * len(anchors))
        if score > best_score:
            best_score = score
            best_sentence = sentence
    return best_score, best_sentence


def compute_risk_signals(row: dict[str, str]) -> RiskSignals:
    anchors = anchor_terms(row["finding_category"], row["finding_definition"])
    s_neg, neg_evidence = negation_score(row["report"], anchors)
    s_imp, imp_evidence = implicit_score(row["report"], anchors)
    return RiskSignals(
        s_neg=round(s_neg, 6),
        s_imp=round(s_imp, 6),
        neg_evidence=neg_evidence,
        imp_evidence=imp_evidence,
    )


def label_text(value: str) -> str:
    return "positive" if str(value).strip() == "1" else "negative"


def correctness(row: dict[str, str]) -> str:
    return "correct" if str(row["audit_label"]).strip() == "1" else "incorrect"


def gold_error_type(row: dict[str, str]) -> str:
    return ERROR_TYPE[row["corruption_type"]]


def choose_evidence(row: dict[str, str], risk: RiskSignals) -> str:
    if row["corruption_type"] == "negative_to_positive" and risk.neg_evidence:
        return risk.neg_evidence
    if risk.imp_evidence:
        return risk.imp_evidence
    if risk.neg_evidence:
        return risk.neg_evidence
    return first_meaningful_sentence(row["report"])[:240]


def first_meaningful_sentence(report: str) -> str:
    for sentence in split_sentences(report):
        tokens = tokenize(sentence)
        if len(tokens) >= 3:
            return sentence
    return split_sentences(report)[0]


def rationale(row: dict[str, str], risk: RiskSignals) -> str:
    finding = row["finding_category"]
    assigned = row["assigned_label_text"]
    reference = label_text(row["reference_label"])
    is_correct = correctness(row) == "correct"

    if is_correct and assigned == "positive":
        return f"The report provides evidence compatible with {finding}, so the positive assigned label is consistent."
    if is_correct and assigned == "negative":
        return f"The report does not provide evidence of {finding}, so the negative assigned label is consistent."
    if reference == "negative" and assigned == "positive":
        return f"The report does not provide evidence of {finding}, so the positive assigned label is inconsistent."
    return f"The report provides evidence compatible with {finding}, so the negative assigned label is inconsistent."


def audit_json(*, correctness_value: str, error_type_value: str, rationale_value: str | None = None) -> str:
    payload = {
        "correctness": correctness_value,
        "error_type": error_type_value,
    }
    if rationale_value:
        payload["rationale"] = rationale_value
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def input_text(row: dict[str, str]) -> str:
    return "\n".join(
        [
            f"Report: {row['report']}",
            f"Finding: {row['finding_category']}",
            f"Definition: {row['finding_definition']}",
            f"Assigned label: {row['assigned_label_text']}",
        ]
    )


def load_rationale_cache(path: Path | None) -> dict[str, str]:
    if path is not None:
        path = resolve_project_path(path)
    if path is None or not path.exists():
        return {}

    rationales: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            sample_id = item.get("sample_id")
            rationale_value = item.get("rationale")
            if not sample_id or not rationale_value:
                raise ValueError(f"{path}:{line_no} must contain sample_id and rationale")
            rationales[str(sample_id)] = str(rationale_value).strip()
    return rationales


def fallback_rationale(row: dict[str, str], risk: RiskSignals, fallback: str) -> str | None:
    if fallback == "minimal":
        return None
    if fallback == "template":
        return rationale(row, risk)
    raise ValueError(f"Unsupported fallback: {fallback}")


def gold_output(
    row: dict[str, str],
    risk: RiskSignals,
    rationale_cache: dict[str, str],
    fallback: str,
) -> str:
    rationale_value = rationale_cache.get(row["sample_id"])
    if rationale_value is None:
        rationale_value = fallback_rationale(row, risk, fallback)
    return audit_json(
        correctness_value=correctness(row),
        error_type_value=gold_error_type(row),
        rationale_value=rationale_value,
    )


def rejected_outputs(
    row: dict[str, str],
    risk: RiskSignals,
    include_rationale: bool,
    include_rationale_mismatch: bool,
) -> list[tuple[str, str]]:
    finding = row["finding_category"]
    corr = correctness(row)
    err = gold_error_type(row)
    assigned = row["assigned_label_text"]
    rationale_or_none = (lambda text: text if include_rationale else None)

    outputs: list[tuple[str, str]] = []
    if corr == "incorrect":
        outputs.append(
            (
                "decision_flip",
                audit_json(
                    correctness_value="correct",
                    error_type_value="none",
                    rationale_value=rationale_or_none(
                        f"Report evidence is treated as sufficient for {finding}, incorrectly supporting the {assigned} label."
                    ),
                ),
            )
        )
        confused = "false_negative" if err == "false_positive" else "false_positive"
        confused_label = "positive" if confused == "false_positive" else "negative"
        outputs.append(
            (
                "error_type_confusion",
                audit_json(
                    correctness_value="incorrect",
                    error_type_value=confused,
                    rationale_value=rationale_or_none(
                        f"Evidence for {finding} is assigned to the wrong error direction, producing a {confused_label} error."
                    ),
                ),
            )
        )
    else:
        wrong_err = "false_positive" if assigned == "positive" else "false_negative"
        outputs.append(
            (
                "false_alarm",
                audit_json(
                    correctness_value="incorrect",
                    error_type_value=wrong_err,
                    rationale_value=rationale_or_none(
                        f"Report evidence is treated as insufficient for {finding}, incorrectly flagging the {assigned} label."
                    ),
                ),
            )
        )

    if include_rationale and include_rationale_mismatch:
        outputs.append(
            (
                "rationale_mismatch",
                audit_json(
                    correctness_value=correctness(row),
                    error_type_value=gold_error_type(row),
                    rationale_value=(
                        f"Evidence unrelated to {finding} is treated as decisive for the audit decision."
                    ),
                ),
            )
        )
    return outputs


def read_rows(path: Path) -> list[dict[str, str]]:
    path = resolve_project_path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_examples(
    rows: list[dict[str, str]],
    rationale_cache: dict[str, str],
    fallback: str,
    require_rationale: bool,
    dpo_with_rationale: bool,
    include_rationale_mismatch: bool,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]], list[dict]]:
    sft_by_split: dict[str, list[dict]] = defaultdict(list)
    dpo_by_split: dict[str, list[dict]] = defaultdict(list)
    risk_rows: list[dict] = []

    for row in rows:
        split = row["split"]
        risk = compute_risk_signals(row)
        if require_rationale and row["sample_id"] not in rationale_cache:
            raise ValueError(f"Missing Qwen rationale for sample_id={row['sample_id']}")
        chosen = gold_output(row, risk, rationale_cache, fallback)
        dpo_chosen = (
            chosen
            if dpo_with_rationale
            else audit_json(correctness_value=correctness(row), error_type_value=gold_error_type(row))
        )
        include_dpo_rationale = dpo_with_rationale and "rationale" in json.loads(chosen)
        common = {
            "instruction": INSTRUCTION,
            "input": input_text(row),
            "sample_id": row["sample_id"],
            "study_id": row["study_id"],
            "finding_category": row["finding_category"],
            "assigned_label_text": row["assigned_label_text"],
            "reference_label_text": label_text(row["reference_label"]),
            "gold_correctness": correctness(row),
            "gold_error_type": gold_error_type(row),
            "error_mode": ERROR_MODE[row["corruption_type"]],
            "s_neg": risk.s_neg,
            "s_imp": risk.s_imp,
            "split": split,
        }

        sft_by_split[split].append({**common, "output": chosen})

        for rejected_type, rejected in rejected_outputs(
            row,
            risk,
            include_rationale=include_dpo_rationale,
            include_rationale_mismatch=include_rationale_mismatch,
        ):
            dpo_by_split[split].append(
                {
                    **common,
                    "chosen": dpo_chosen,
                    "rejected": rejected,
                    "rejected_type": rejected_type,
                }
            )

        risk_rows.append(
            {
                **common,
                "corruption_type": row["corruption_type"],
                "neg_evidence": risk.neg_evidence,
                "imp_evidence": risk.imp_evidence,
                "qwen_rationale": rationale_cache.get(row["sample_id"], ""),
            }
        )

    return dict(sft_by_split), dict(dpo_by_split), risk_rows


def dataset_info() -> dict:
    return {
        "ranaudit_sft_train": {
            "file_name": "ranaudit_sft_train.jsonl",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        },
        "ranaudit_sft_val": {
            "file_name": "ranaudit_sft_val.jsonl",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        },
        "ranaudit_sft_test": {
            "file_name": "ranaudit_sft_test.jsonl",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        },
        "ranaudit_dpo_train": {
            "file_name": "ranaudit_dpo_train.jsonl",
            "formatting": "alpaca",
            "ranking": True,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "chosen": "chosen",
                "rejected": "rejected",
            },
        },
        "ranaudit_dpo_val": {
            "file_name": "ranaudit_dpo_val.jsonl",
            "formatting": "alpaca",
            "ranking": True,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "chosen": "chosen",
                "rejected": "rejected",
            },
        },
        "ranaudit_dpo_test": {
            "file_name": "ranaudit_dpo_test.jsonl",
            "formatting": "alpaca",
            "ranking": True,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "chosen": "chosen",
                "rejected": "rejected",
            },
        },
    }


def summarize(
    rows: list[dict[str, str]],
    sft_by_split: dict[str, list[dict]],
    dpo_by_split: dict[str, list[dict]],
    risk_rows: list[dict],
    dpo_with_rationale: bool,
    include_rationale_mismatch: bool,
) -> dict:
    split_counts = Counter(row["split"] for row in rows)
    finding_counts = Counter(row["finding_category"] for row in rows)
    corruption_counts = Counter(row["corruption_type"] for row in rows)
    rejected_counts = Counter(row["rejected_type"] for split_rows in dpo_by_split.values() for row in split_rows)

    risk_by_mode: dict[str, dict[str, float]] = {}
    for mode in sorted({row["error_mode"] for row in risk_rows}):
        mode_rows = [row for row in risk_rows if row["error_mode"] == mode]
        risk_by_mode[mode] = {
            "count": len(mode_rows),
            "mean_s_neg": round(mean(row["s_neg"] for row in mode_rows), 6),
            "mean_s_imp": round(mean(row["s_imp"] for row in mode_rows), 6),
        }

    return {
        "input_rows": len(rows),
        "split_counts": dict(split_counts),
        "finding_counts": dict(finding_counts),
        "corruption_counts": dict(corruption_counts),
        "sft_counts": {split: len(split_rows) for split, split_rows in sorted(sft_by_split.items())},
        "dpo_counts": {split: len(split_rows) for split, split_rows in sorted(dpo_by_split.items())},
        "rejected_type_counts": dict(rejected_counts),
        "dpo_with_rationale": dpo_with_rationale,
        "include_rationale_mismatch": include_rationale_mismatch,
        "risk_by_mode": risk_by_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build RanAudit SFT/DPO data for LlamaFactory and ALSPC experiments."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--rationale-cache",
        type=Path,
        default=DEFAULT_RATIONALE_CACHE,
        help="JSONL cache generated by generate_qwen_rationales.py.",
    )
    parser.add_argument(
        "--fallback",
        choices=("minimal", "template"),
        default="minimal",
        help="What to do when a sample is missing from the rationale cache.",
    )
    parser.add_argument(
        "--require-rationale",
        action="store_true",
        help="Fail if any sample is missing a Qwen-generated rationale.",
    )
    parser.add_argument(
        "--dpo-with-rationale",
        action="store_true",
        help=(
            "Include rationale in DPO chosen/rejected payloads. By default, DPO uses a minimal structured fallback "
            "to avoid teaching style preferences from synthetic rejected rationales."
        ),
    )
    parser.add_argument(
        "--include-rationale-mismatch",
        action="store_true",
        help="Also build rationale_mismatch rejected samples. Disabled by default to reduce DPO noise.",
    )
    args = parser.parse_args()

    args.output_dir = resolve_project_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.input)
    rationale_cache = load_rationale_cache(args.rationale_cache)
    sft_by_split, dpo_by_split, risk_rows = build_examples(
        rows=rows,
        rationale_cache=rationale_cache,
        fallback=args.fallback,
        require_rationale=args.require_rationale,
        dpo_with_rationale=args.dpo_with_rationale,
        include_rationale_mismatch=args.include_rationale_mismatch,
    )

    for split, split_rows in sorted(sft_by_split.items()):
        write_jsonl(args.output_dir / f"ranaudit_sft_{split}.jsonl", split_rows)
    for split, split_rows in sorted(dpo_by_split.items()):
        write_jsonl(args.output_dir / f"ranaudit_dpo_{split}.jsonl", split_rows)

    write_jsonl(args.output_dir / "ranaudit_alspc_risk_signals.jsonl", risk_rows)

    with (args.output_dir / "dataset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_info(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    summary = summarize(
        rows,
        sft_by_split,
        dpo_by_split,
        risk_rows,
        dpo_with_rationale=args.dpo_with_rationale,
        include_rationale_mismatch=args.include_rationale_mismatch,
    )
    with (args.output_dir / "build_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
