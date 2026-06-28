from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from build_llamafactory_data import ERROR_TYPE, correctness, gold_error_type, input_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path(
    "data/processed/RanAudit_binary_label_consistency_dataset_balanced_8400_study_split.csv"
)
DEFAULT_OUTPUT = Path("data/llamafactory/qwen_rationales.jsonl")
DEFAULT_MODEL = "qwen-plus"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


SYSTEM_PROMPT = (
    "You are an expert clinical radiology auditor. "
    "The fixed gold correctness and error type must not be changed. "
    "Your only job is to write one concise, highly specific, report-grounded rationale. "
    "Use conservative audit language and avoid boilerplate wording."
)


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_rows(path: Path) -> list[dict[str, str]]:
    path = resolve_project_path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_done(path: Path) -> set[str]:
    path = resolve_project_path(path)
    if not path.exists():
        return set()

    done = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("sample_id") and item.get("rationale"):
                done.add(str(item["sample_id"]))
    return done


def rationale_guidance(row: dict[str, str]) -> str:
    corr = correctness(row)
    err = gold_error_type(row)
    assigned = row["assigned_label_text"].lower()
    finding = row["finding_category"]

    if corr == "correct" and assigned == "positive":
        return f"The report supports {finding}; say that the positive label is supported or correct."
    if corr == "correct" and assigned == "negative":
        return f"The report lacks support for {finding}; say that the negative label is supported, appropriate, or correct."
    if err == "false_positive":
        return f"The report does not support {finding}; say that the positive label is unsupported or false positive."
    if err == "false_negative":
        return f"The report supports {finding}; say that the negative label is incorrect or misses a supported finding."
    return "Explain the fixed audit result using only report evidence."


def build_prompt(row: dict[str, str]) -> str:
    gold_payload = {
        "correctness": correctness(row),
        "error_type": gold_error_type(row),
    }
    assigned = row["assigned_label_text"].lower()
    target_guidance = rationale_guidance(row)

    return (
        "Input case:\n"
        f"{input_text(row)}\n\n"
        "Fixed gold audit result:\n"
        f"{json.dumps(gold_payload, ensure_ascii=False)}\n\n"
        f"Interpretation guidance: {target_guidance}\n\n"
        "Write exactly ONE concise English sentence as the rationale. Strict requirements:\n"
        "- Do not change correctness or error_type.\n"
        "- Start directly with concrete radiographic evidence, not with a generic prefix. "
        "Do not begin with phrases such as 'The assigned label', 'The label', "
        "'The report', 'This report', or 'There is'.\n"
        "- Use only this audit vocabulary: supported, unsupported, correct, incorrect, "
        "false positive, false negative, or misses a supported finding.\n"
        "- Do not use the words contradict, contradicted, contradiction, conflict, negates, or negated.\n"
        "- For a correct negative label, say the negative label is supported, appropriate, or correct; "
        "never say it is unsupported.\n"
        "- Do not recite textbook disease definitions or generic disease features unless they explicitly appear in the report.\n"
        "- Do not quote report phrases. Do not use single or double quotation marks inside the rationale; "
        "paraphrase concrete report evidence instead.\n"
        "- Keep it under 35 words.\n"
        f"- The sentence must naturally include the word '{assigned}' once to indicate assigned label polarity, "
        "but do not force a repeated ending template.\n"
        "Good style examples:\n"
        "- Clear lungs and no acute cardiopulmonary process leave the positive atelectasis label unsupported.\n"
        "- Bibasilar atelectasis is explicitly described, so the negative label misses a supported finding.\n"
        "- Normal heart size and normal mediastinal contours support the negative enlarged-cardiomediastinum label.\n"
        "- Output JSON only with exactly this schema: "
        "{\"rationale\":\"...\"}"
    )


def retry_prompt(row: dict[str, str], error: Exception) -> str:
    base_prompt = build_prompt(row)
    reason = str(error)
    if reason == "rationale is too long":
        return (
            f"{base_prompt}\n\n"
            "The previous draft was rejected because it was too long. "
            "Rewrite using 18-25 words only. Use exactly one core report evidence phrase, "
            "then one audit conclusion. Do not list multiple absent findings."
        )
    if reason == "false positive rationale may imply positive support":
        return (
            f"{base_prompt}\n\n"
            "The previous draft was rejected because it sounded as if the positive label was supported. "
            "Rewrite as: one absent or alternative finding, then 'so the positive label is unsupported'. "
            "Do not write 'supports the positive label' or 'support the positive label'."
        )
    if reason == "banned contradiction-style wording":
        return (
            f"{base_prompt}\n\n"
            "The previous draft used banned contradiction-style wording. "
            "Rewrite without contradict, contradiction, conflict, negates, or negated. "
            "Use 'unsupported' or 'supported' instead."
        )
    if reason == "rationale contains quotation marks":
        return (
            f"{base_prompt}\n\n"
            "The previous draft used quotation marks. Rewrite without any single or double quotes. "
            "Paraphrase the report evidence in plain language."
        )
    if reason == "generic disease-definition phrase detected":
        return (
            f"{base_prompt}\n\n"
            "The previous draft sounded like a textbook definition. "
            "Rewrite using only this specific report's observation and the assigned label conclusion. "
            "Do not mention generic disease features."
        )
    if reason == "false negative rationale says negative label is correct":
        return (
            f"{base_prompt}\n\n"
            "The previous draft incorrectly described the negative label as correct. "
            "Rewrite so the negative label is incorrect or misses a supported finding."
        )
    return (
        f"{base_prompt}\n\n"
        f"The previous draft was rejected because: {reason}. "
        "Rewrite it while preserving the fixed gold result and avoiding that error."
    )


def call_qwen(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout: int,
) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def parse_rationale(content: str) -> str:
    payload = json.loads(content)
    rationale = str(payload.get("rationale", "")).strip()
    if not rationale:
        raise ValueError("Qwen response has empty rationale")
    return clean_rationale(rationale)


def clean_rationale(rationale: str) -> str:
    rationale = rationale.strip()
    rationale = re.sub(r"\s+", " ", rationale)
    rationale = (
        rationale.replace("\u2014", ", ")
        .replace("\u2013", ", ")
        .replace("\u9225\u650e", ", m")
        .replace("\u9225\u6515", ", s")
        .replace("\u9225\u6506", ", f")
        .replace("\u9225\u6501", ", a")
        .replace("\u9225\u6516", ", t")
        .replace("\u9225?", ", ")
    )

    prefix_patterns = [
        r"^The report (?:explicitly )?(?:states|says|describes|notes|mentions|shows|identifies)\s+(?:that\s+)?",
        r"^This report (?:explicitly )?(?:states|says|describes|notes|mentions|shows|identifies)\s+(?:that\s+)?",
        r"^The assigned label ['\"]?(?:positive|negative)['\"]?(?: for [^ ]+)? is (?:incorrect|correct) because\s+",
        r"^The label is (?:incorrect|correct) because\s+",
        r"^There (?:is|are)\s+",
    ]
    for pattern in prefix_patterns:
        rationale = re.sub(pattern, "", rationale, flags=re.IGNORECASE).strip()

    if rationale:
        rationale = rationale[0].upper() + rationale[1:]
    return rationale


def validate_rationale(row: dict[str, str], rationale: str) -> None:
    rationale = clean_rationale(rationale)
    lowered = rationale.lower()

    if len(rationale.split()) < 6:
        raise ValueError("rationale is too short")

    if len(rationale.split()) > 35:
        raise ValueError("rationale is too long")

    if "'" in rationale or '"' in rationale:
        raise ValueError("rationale contains quotation marks")

    banned_starts = [
        "the assigned label",
        "this assigned label",
        "assigned label",
        "the report",
        "this report",
        "the label is",
        "there is",
        "there are",
    ]
    if any(lowered.startswith(x) for x in banned_starts):
        raise ValueError("template-like rationale opening")

    assigned = row["assigned_label_text"].lower()
    if assigned not in lowered:
        raise ValueError("rationale does not mention assigned label polarity")

    generic_phrases = [
        "typically manifest",
        "classic radiographic signs",
        "required for",
        "defined as",
        "textbook",
        "disease features",
        "features suggestive of",
        "features supporting",
        "features required",
    ]
    if any(x in lowered for x in generic_phrases):
        raise ValueError("generic disease-definition phrase detected")

    banned_reasoning_terms = [
        r"\bcontradict(?:s|ed|ing|ion|ory)?\b",
        r"\bconflict(?:s|ed|ing)?\b",
        r"\bnegat(?:es|ed|ing|ion|ions)\b",
    ]
    if any(re.search(pattern, lowered) for pattern in banned_reasoning_terms):
        raise ValueError("banned contradiction-style wording")

    corr = correctness(row)
    err = gold_error_type(row)

    if corr == "correct" and assigned == "negative":
        bad_phrases = [
            "negative label unsupported",
            "negative label is unsupported",
            "negative label remains unsupported",
            "negative edema label unsupported",
            "negative atelectasis label unsupported",
            "leave the negative label unsupported",
            "leaves the negative label unsupported",
            "leaving the negative label unsupported",
        ]
        if any(phrase in lowered for phrase in bad_phrases):
            raise ValueError("correct negative label described as unsupported")

    if err == "false_positive" and "positive" in lowered and "support" in lowered:
        if (
            "unsupported" not in lowered
            and "not support" not in lowered
            and "not supported" not in lowered
            and "does not support" not in lowered
        ):
            raise ValueError("false positive rationale may imply positive support")

    if err == "false_negative" and re.search(r"\bnegative\b", lowered) and re.search(
        r"\b(?:is|label is|label remains)\s+correct\b", lowered
    ):
        raise ValueError("false negative rationale says negative label is correct")


def write_record(path: Path, record: dict) -> None:
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Qwen rationales while locking RanAudit gold labels."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()
    args.output = resolve_project_path(args.output)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key. Set ${args.api_key_env} before running this script.")

    rows = read_rows(args.input)
    if args.split:
        rows = [row for row in rows if row["split"] == args.split]
    if args.limit is not None:
        rows = rows[: args.limit]

    done = load_done(args.output)
    pending = [row for row in rows if row["sample_id"] not in done]

    for index, row in enumerate(pending, start=1):
        prompt = build_prompt(row)
        for attempt in range(1, 4):
            try:
                content = call_qwen(
                    api_key=api_key,
                    base_url=args.base_url,
                    model=args.model,
                    prompt=prompt,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
                rationale = parse_rationale(content)
                validate_rationale(row, rationale)
                write_record(
                    args.output,
                    {
                        "sample_id": row["sample_id"],
                        "split": row["split"],
                        "finding_category": row["finding_category"],
                        "assigned_label_text": row["assigned_label_text"],
                        "gold_correctness": correctness(row),
                        "gold_error_type": ERROR_TYPE[row["corruption_type"]],
                        "model": args.model,
                        "rationale": rationale,
                    },
                )
                print(f"[{index}/{len(pending)}] OK {row['sample_id']}")
                break
            except (ValueError, urllib.error.URLError, TimeoutError) as exc:
                if attempt == 3:
                    failed_path = args.output.with_name(f"{args.output.stem}.failed.jsonl")
                    write_record(
                        failed_path,
                        {
                            "sample_id": row["sample_id"],
                            "split": row["split"],
                            "finding_category": row["finding_category"],
                            "assigned_label_text": row["assigned_label_text"],
                            "error": str(exc),
                        },
                    )
                    print(f"[{index}/{len(pending)}] SKIP {row['sample_id']} | {exc}")
                    break
                prompt = retry_prompt(row, exc)
                time.sleep(args.sleep * attempt)
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
