from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


DEFAULT_LABEL_COLUMNS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Lung Opacity",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
    "emphysema",
    "interstitial lung disease",
    "calcification(lung and mediastinal)",
    "Trachea and bronchus",
    "cavity and cyst",
    "mediastinal other",
    "pulmonary vascular abnormal",
]


def split_columns(value: str | None) -> list[str]:
    if not value:
        return DEFAULT_LABEL_COLUMNS
    return [item.strip() for item in value.split(",") if item.strip()]


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def coerce_binary(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).clip(lower=0, upper=1)


def evaluate(gold: pd.DataFrame, pred: pd.DataFrame, label_columns: list[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    missing_gold = [column for column in label_columns if column not in gold.columns]
    missing_pred = [column for column in label_columns if column not in pred.columns]
    if missing_gold:
        raise SystemExit(f"Missing label columns in gold file: {missing_gold}")
    if missing_pred:
        raise SystemExit(f"Missing label columns in prediction file: {missing_pred}")
    if len(gold) != len(pred):
        raise SystemExit(f"Gold and prediction row counts differ: {len(gold)} vs {len(pred)}")

    rows = []
    y_true_all = []
    y_pred_all = []
    for label in label_columns:
        y_true = coerce_binary(gold[label])
        y_pred = coerce_binary(pred[label])
        y_true_all.extend(y_true.tolist())
        y_pred_all.extend(y_pred.tolist())
        rows.append(
            {
                "label": label,
                "n_positive": int(y_true.sum()),
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }
        )

    per_label = pd.DataFrame(rows)
    summary = {
        "n_reports": float(len(gold)),
        "n_labels": float(len(label_columns)),
        "macro_precision": float(per_label["precision"].mean()),
        "macro_recall": float(per_label["recall"].mean()),
        "macro_f1": float(per_label["f1"].mean()),
        "micro_precision": float(precision_score(y_true_all, y_pred_all, zero_division=0)),
        "micro_recall": float(recall_score(y_true_all, y_pred_all, zero_division=0)),
        "micro_f1": float(f1_score(y_true_all, y_pred_all, zero_division=0)),
    }
    return per_label, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Ran Score finding labels against a reference label table.")
    parser.add_argument("--gold", required=True, type=Path, help="Reference CSV/XLSX with binary finding labels.")
    parser.add_argument("--pred", required=True, type=Path, help="Prediction CSV/XLSX with binary finding labels.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for per-label and summary metrics.")
    parser.add_argument("--label-columns", default=None, help="Optional comma-separated label column list.")
    args = parser.parse_args()

    labels = split_columns(args.label_columns)
    gold = load_table(args.gold)
    pred = load_table(args.pred)
    per_label, summary = evaluate(gold, pred, labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_label.to_csv(args.output_dir / "per_label_metrics.csv", index=False, float_format="%.6f")
    pd.DataFrame([summary]).to_csv(args.output_dir / "summary_metrics.csv", index=False, float_format="%.6f")

    print("Ran Score evaluation summary")
    for key, value in summary.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
