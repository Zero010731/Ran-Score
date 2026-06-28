import argparse
import json
import math
import re
import shutil
from pathlib import Path


SPLITS = ("train", "val", "test")
UNCERTAINTY_PATTERNS = (
    r"\bcould be\b",
    r"\bmay reflect\b",
    r"\bmay represent\b",
    r"\bpossibly\b",
    r"\bpossible\b",
    r"\bcannot exclude\b",
    r"\bdifferential\b",
    r"\bversus\b",
    r"\bvs\.?\b",
)


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


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def percentile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[int(pos)]
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def minmax_stats(rows, field, lower_q, upper_q):
    values = [float(row.get(field, 0.0) or 0.0) for row in rows]
    lo = percentile(values, lower_q)
    hi = percentile(values, upper_q)
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def normalize(value, lo, hi):
    value = float(value or 0.0)
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def softmax2(a, b, tau):
    tau = max(float(tau), 1e-6)
    ea = math.exp(a / tau)
    eb = math.exp(b / tau)
    denom = ea + eb
    return ea / denom, eb / denom


def get_prompt(row):
    if row.get("input"):
        return str(row["input"])
    if row.get("instruction"):
        return f"{row.get('instruction', '')}\n{row.get('input', '')}".strip()
    if row.get("prompt"):
        return str(row["prompt"])
    return json.dumps(row, ensure_ascii=False)


def extract_field(prompt, name):
    match = re.search(rf"{re.escape(name)}:\s*(.+)", prompt)
    return match.group(1).strip() if match else ""


def extract_report(prompt):
    match = re.search(r"Report:\s*(.*?)(?:\nFinding:|\sFinding:)", prompt, flags=re.S)
    return match.group(1).strip() if match else prompt


def uncertainty_positive_signal(row):
    prompt = get_prompt(row)
    assigned_label = extract_field(prompt, "Assigned label") or str(row.get("assigned_label_text", ""))
    if assigned_label.lower() != "positive":
        return 0.0, ""

    report = extract_report(prompt).lower()
    for pattern in UNCERTAINTY_PATTERNS:
        match = re.search(pattern, report)
        if match:
            return 1.0, match.group(0)
    return 0.0, ""


def compute_margin(row, stats, args):
    s_neg = normalize(row.get("s_neg", 0.0), *stats["s_neg"])
    s_imp = normalize(row.get("s_imp", 0.0), *stats["s_imp"])
    g_neg, g_imp = softmax2(s_neg, s_imp, args.tau)
    s_unc, unc_hit = uncertainty_positive_signal(row)

    mode = row.get("error_mode")
    margin = args.m0
    if args.mode == "fixed":
        margin = args.fixed_margin
    elif mode == "FP":
        margin += args.alpha_neg * g_neg * s_neg
        if args.mode == "alspc_uncertainty":
            margin += args.alpha_unc * s_unc
    elif mode == "FN":
        margin += args.alpha_imp * g_imp * s_imp

    margin = max(args.m_min, min(args.m_max, margin))
    return {
        "alspc_margin": round(margin, 6),
        "alspc_s_neg_norm": round(s_neg, 6),
        "alspc_s_imp_norm": round(s_imp, 6),
        "alspc_g_neg": round(g_neg, 6),
        "alspc_g_imp": round(g_imp, 6),
        "alspc_s_uncertainty_positive": round(s_unc, 6),
        "alspc_uncertainty_hit": unc_hit,
    }


def summarize(rows):
    by_mode = {}
    for row in rows:
        mode = row.get("error_mode", "NA")
        item = by_mode.setdefault(mode, {"n": 0, "sum": 0.0, "min": None, "max": None})
        value = float(row["alspc_margin"])
        item["n"] += 1
        item["sum"] += value
        item["min"] = value if item["min"] is None else min(item["min"], value)
        item["max"] = value if item["max"] is None else max(item["max"], value)
    for item in by_mode.values():
        item["mean"] = round(item["sum"] / item["n"], 6) if item["n"] else 0.0
        item.pop("sum", None)
    return by_mode


def main():
    parser = argparse.ArgumentParser(description="Build ALSPC-margin DPO data from RanAudit DPO JSONL files.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["alspc", "alspc_uncertainty", "fixed"], default="alspc")
    parser.add_argument("--m0", type=float, default=0.1)
    parser.add_argument("--alpha-neg", type=float, default=1.0)
    parser.add_argument("--alpha-imp", type=float, default=1.0)
    parser.add_argument("--alpha-unc", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--m-min", type=float, default=0.0)
    parser.add_argument("--m-max", type=float, default=0.5)
    parser.add_argument("--fixed-margin", type=float, default=0.5)
    parser.add_argument("--lower-quantile", type=float, default=0.01)
    parser.add_argument("--upper-quantile", type=float, default=0.99)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(input_dir / "ranaudit_dpo_train.jsonl")
    stats = {
        "s_neg": minmax_stats(train_rows, "s_neg", args.lower_quantile, args.upper_quantile),
        "s_imp": minmax_stats(train_rows, "s_imp", args.lower_quantile, args.upper_quantile),
    }

    summary = {
        "mode": args.mode,
        "params": {
            "m0": args.m0,
            "alpha_neg": args.alpha_neg,
            "alpha_imp": args.alpha_imp,
            "alpha_unc": args.alpha_unc,
            "tau": args.tau,
            "m_min": args.m_min,
            "m_max": args.m_max,
            "fixed_margin": args.fixed_margin,
            "lower_quantile": args.lower_quantile,
            "upper_quantile": args.upper_quantile,
        },
        "normalization": "train quantile-clipped min-max",
        "normalization_stats_from_train": stats,
        "splits": {},
    }

    for split in SPLITS:
        in_path = input_dir / f"ranaudit_dpo_{split}.jsonl"
        rows = read_jsonl(in_path)
        out_rows = []
        for row in rows:
            new_row = dict(row)
            new_row.update(compute_margin(row, stats, args))
            out_rows.append(new_row)
        write_jsonl(output_dir / f"ranaudit_dpo_{split}.jsonl", out_rows)
        summary["splits"][split] = {
            "n": len(out_rows),
            "margin_by_error_mode": summarize(out_rows),
        }

    for name in ["ranaudit_sft_train.jsonl", "ranaudit_sft_val.jsonl", "ranaudit_sft_test.jsonl"]:
        src = input_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    for name in ["dataset_info.json", "build_summary.json", "ranaudit_alspc_risk_signals.jsonl"]:
        src = input_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    (output_dir / "alspc_margin_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
