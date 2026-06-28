#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train a Hugging Face sequence classifier for the label-audit task.

This script is a general version of the BERT audit baseline trainer. It supports
AutoModelForSequenceClassification models such as BERT, RoBERTa, ELECTRA, and
Longformer, while reporting the same task metrics used in the audit experiments.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = ["none", "false_positive", "false_negative"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


TEXT_KEYS = [
    "text",
    "input_text",
    "source",
    "prompt",
    "input",
    "query",
]

LABEL_KEYS = [
    "label",
    "labels",
    "error_type",
    "gold_error_type",
    "target",
    "answer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", type=float, default=5.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-field", default=None)
    parser.add_argument("--label-field", default=None)
    parser.add_argument("--use-fast-tokenizer", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=50)
    return parser.parse_args()


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for key in ["data", "examples", "records"]:
                if isinstance(obj.get(key), list):
                    return obj[key]
        raise ValueError(f"Unsupported JSON structure in {path}")

    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def read_table(path: Path) -> list[dict[str, Any]]:
    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def find_split_file(data_dir: Path, split: str) -> Path | None:
    names_by_split = {
        "train": ["train"],
        "val": ["val", "valid", "validation", "dev"],
        "test": ["test"],
    }
    exts = [".jsonl", ".json", ".csv", ".tsv"]
    for name in names_by_split[split]:
        for ext in exts:
            path = data_dir / f"{name}{ext}"
            if path.exists():
                return path
    return None


def load_split(path: Path) -> list[dict[str, Any]]:
    if path.suffix in {".json", ".jsonl"}:
        return read_json_or_jsonl(path)
    if path.suffix in {".csv", ".tsv"}:
        return read_table(path)
    raise ValueError(f"Unsupported data file extension: {path}")


def normalize_label(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("Boolean labels are not supported.")
    if isinstance(value, int):
        if value in ID2LABEL:
            return ID2LABEL[value]
        raise ValueError(f"Unknown numeric label: {value}")
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return normalize_label(int(value))

    s = str(value).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    aliases = {
        "0": "none",
        "1": "false_positive",
        "2": "false_negative",
        "normal": "none",
        "correct": "none",
        "none": "none",
        "no_error": "none",
        "false_positive": "false_positive",
        "fp": "false_positive",
        "false_pos": "false_positive",
        "false_negative": "false_negative",
        "fn": "false_negative",
        "false_neg": "false_negative",
    }
    if s in aliases:
        return aliases[s]
    raise ValueError(f"Unknown label: {value!r}")


def maybe_parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s or s[0] not in "[{":
        return value
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return value


def extract_label(row: dict[str, Any], label_field: str | None = None) -> str:
    if label_field:
        if label_field not in row:
            raise KeyError(f"Missing label field {label_field!r}")
        return normalize_label(row[label_field])

    for key in LABEL_KEYS:
        if key in row and row[key] not in (None, ""):
            value = maybe_parse_json(row[key])
            if isinstance(value, dict) and "error_type" in value:
                return normalize_label(value["error_type"])
            return normalize_label(value)

    for key in ["output", "response", "completion", "target_text"]:
        value = maybe_parse_json(row.get(key))
        if isinstance(value, dict) and "error_type" in value:
            return normalize_label(value["error_type"])

    if row.get("correctness") == "correct":
        return "none"

    raise KeyError(f"Could not find label in record keys: {sorted(row.keys())}")


def stringify_messages(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return None
    parts = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or msg.get("from") or "message"
        content = msg.get("content") or msg.get("value") or ""
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts) if parts else None


def extract_text(row: dict[str, Any], text_field: str | None = None) -> str:
    if text_field:
        if text_field not in row:
            raise KeyError(f"Missing text field {text_field!r}")
        return str(row[text_field])

    for key in TEXT_KEYS:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)

    for key in ["messages", "conversations"]:
        value = stringify_messages(row.get(key))
        if value:
            return value

    pieces = []
    named_fields = [
        ("Report", ["report", "radiology_report", "findings_text", "findings", "impression"]),
        ("Finding", ["finding", "finding_name", "finding_category"]),
        ("Definition", ["definition", "finding_definition", "label_definition"]),
        ("Assigned label", ["assigned_label", "assigned_label_text", "candidate_label"]),
    ]
    for title, keys in named_fields:
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                pieces.append(f"{title}: {value}")
                break

    if pieces:
        return "\n".join(pieces)

    raise KeyError(f"Could not find text in record keys: {sorted(row.keys())}")


class AuditDataset:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
        text_field: str | None,
        label_field: str | None,
        add_longformer_global_attention: bool,
    ) -> None:
        self.records = []
        for idx, row in enumerate(rows):
            text = extract_text(row, text_field)
            label = extract_label(row, label_field)
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            if add_longformer_global_attention:
                attention = [0] * len(encoded["input_ids"])
                if attention:
                    attention[0] = 1
                encoded["global_attention_mask"] = attention
            encoded["labels"] = LABEL2ID[label]
            self.records.append(
                {
                    "features": encoded,
                    "meta": {
                        "idx": idx,
                        "sample_id": row.get("sample_id") or row.get("id") or row.get("uid"),
                        "gold": label,
                        "text": text,
                    },
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return dict(self.records[idx]["features"])


class AuditDataCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        import torch

        global_masks = [feature.pop("global_attention_mask", None) for feature in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")

        if any(mask is not None for mask in global_masks):
            seq_len = int(batch["input_ids"].shape[1])
            padded_masks = []
            for mask in global_masks:
                if mask is None:
                    mask = [0]
                mask = list(mask[:seq_len])
                pad_len = seq_len - len(mask)
                if self.tokenizer.padding_side == "left":
                    mask = [0] * pad_len + mask
                else:
                    mask = mask + [0] * pad_len
                padded_masks.append(mask)
            batch["global_attention_mask"] = torch.tensor(padded_masks, dtype=torch.long)

        return batch


def load_tokenizer(args: argparse.Namespace) -> Any:
    from transformers import AutoTokenizer

    use_fast_values = [True, False] if args.use_fast_tokenizer == "auto" else [args.use_fast_tokenizer == "true"]
    last_error = None
    for use_fast in use_fast_values:
        try:
            return AutoTokenizer.from_pretrained(
                args.model_name_or_path,
                use_fast=use_fast,
                trust_remote_code=args.trust_remote_code,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[tokenizer] AutoTokenizer(use_fast={use_fast}) failed: {exc}")

    model_dir = Path(args.model_name_or_path)
    if (model_dir / "vocab.txt").exists():
        from transformers import BertTokenizer

        lower = "cased" not in model_dir.name.lower() or "uncased" in model_dir.name.lower()
        print(f"[tokenizer] Falling back to BertTokenizer(do_lower_case={lower}).")
        return BertTokenizer.from_pretrained(str(model_dir), do_lower_case=lower)

    raise RuntimeError(f"Could not load tokenizer from {args.model_name_or_path}") from last_error


def try_manual_safe_bin_load(args: argparse.Namespace, config: Any) -> Any:
    import torch
    from transformers import AutoModelForSequenceClassification

    model_dir = Path(args.model_name_or_path)
    bin_path = model_dir / "pytorch_model.bin"
    if not bin_path.exists():
        raise FileNotFoundError(f"No pytorch_model.bin found in {model_dir}")

    print("[model] Falling back to manual torch.load(weights_only=True).")
    model = AutoModelForSequenceClassification.from_config(config)
    state = torch.load(bin_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(state)!r}")

    skip_prefixes = (
        "classifier.",
        "score.",
        "pre_classifier.",
        "cls.",
        "lm_head.",
        "discriminator_predictions.",
        "generator_predictions.",
        "qa_outputs.",
    )
    filtered = {
        key: value
        for key, value in state.items()
        if isinstance(value, torch.Tensor) and not key.startswith(skip_prefixes)
    }
    load_report = model.load_state_dict(filtered, strict=False)
    print(f"[model] loaded tensors: {len(filtered)}")
    print(f"[model] missing keys: {len(load_report.missing_keys)}")
    print(f"[model] unexpected keys: {len(load_report.unexpected_keys)}")
    if load_report.missing_keys:
        print("[model] first missing keys:", load_report.missing_keys[:10])
    if load_report.unexpected_keys:
        print("[model] first unexpected keys:", load_report.unexpected_keys[:10])
    return model


def load_model(args: argparse.Namespace) -> Any:
    from transformers import AutoConfig, AutoModelForSequenceClassification

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        trust_remote_code=args.trust_remote_code,
    )
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            config=config,
            ignore_mismatched_sizes=True,
            trust_remote_code=args.trust_remote_code,
        )
    except ValueError as exc:
        if "torch.load" not in str(exc) and "safetensors" not in str(exc):
            raise
        model = try_manual_safe_bin_load(args, config)

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 4) if den else 0.0


def macro_f1(preds: list[int], labels: list[int]) -> float:
    scores = []
    for label_id in range(len(LABELS)):
        tp = sum(p == label_id and y == label_id for p, y in zip(preds, labels))
        fp = sum(p == label_id and y != label_id for p, y in zip(preds, labels))
        fn = sum(p != label_id and y == label_id for p, y in zip(preds, labels))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)
    return round(100.0 * sum(scores) / len(scores), 4)


def compute_flat_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    pred_ids = logits.argmax(axis=-1).tolist()
    label_ids = labels.tolist()
    correct = sum(p == y for p, y in zip(pred_ids, label_ids))
    return {
        "accuracy": pct(correct, len(label_ids)),
        "joint_accuracy": pct(correct, len(label_ids)),
        "macro_f1": macro_f1(pred_ids, label_ids),
    }


def detailed_metrics(pred_ids: list[int], label_ids: list[int]) -> dict[str, Any]:
    total = len(label_ids)
    correct = sum(p == y for p, y in zip(pred_ids, label_ids))
    gold_names = [ID2LABEL[y] for y in label_ids]
    pred_names = [ID2LABEL[p] for p in pred_ids]

    by_gold: dict[str, dict[str, Any]] = {}
    for label in LABELS:
        idxs = [i for i, gold in enumerate(gold_names) if gold == label]
        by_gold[label] = {
            "n": len(idxs),
            "joint_accuracy": pct(sum(pred_names[i] == label for i in idxs), len(idxs)),
        }

    confusion: dict[str, dict[str, int]] = {}
    for label in LABELS:
        counter = Counter(pred_names[i] for i, gold in enumerate(gold_names) if gold == label)
        confusion[label] = dict(counter)

    none_idxs = [i for i, gold in enumerate(gold_names) if gold == "none"]
    error_idxs = [i for i, gold in enumerate(gold_names) if gold != "none"]
    false_alarm = sum(pred_names[i] != "none" for i in none_idxs)
    missed_error = sum(pred_names[i] == "none" for i in error_idxs)

    return {
        "total": total,
        "accuracy": pct(correct, total),
        "joint_accuracy": pct(correct, total),
        "macro_f1": macro_f1(pred_ids, label_ids),
        "false_alarm_rate_on_none": pct(false_alarm, len(none_idxs)),
        "missed_error_rate_on_error_samples": pct(missed_error, len(error_idxs)),
        "by_gold_error_type": by_gold,
        "confusion_error_type": confusion,
    }


def write_predictions(path: Path, dataset: AuditDataset, pred_ids: list[int]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item, pred_id in zip(dataset.records, pred_ids):
            meta = item["meta"]
            row = {
                "idx": meta["idx"],
                "sample_id": meta["sample_id"],
                "gold_error_type": meta["gold"],
                "pred_error_type": ID2LABEL[pred_id],
                "correct": meta["gold"] == ID2LABEL[pred_id],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_training_args(args: argparse.Namespace) -> TrainingArguments:
    from transformers import TrainingArguments

    params = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "joint_accuracy",
        "greater_is_better": True,
        "report_to": "none",
        "seed": args.seed,
        "fp16": args.fp16,
        "bf16": args.bf16,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    valid = set(signature.parameters)
    if "eval_strategy" in valid:
        params["eval_strategy"] = "epoch"
    else:
        params["evaluation_strategy"] = "epoch"
    if "save_strategy" in valid:
        params["save_strategy"] = "epoch"
    if "logging_strategy" in valid:
        params["logging_strategy"] = "steps"
    return TrainingArguments(**{k: v for k, v in params.items() if k in valid})


def main() -> None:
    args = parse_args()
    from transformers import Trainer, set_seed

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    train_path = find_split_file(data_dir, "train")
    val_path = find_split_file(data_dir, "val")
    test_path = find_split_file(data_dir, "test")
    if train_path is None or val_path is None:
        raise FileNotFoundError(f"Need train and val/validation files under {data_dir}")

    print(f"[data] train: {train_path}")
    print(f"[data] val:   {val_path}")
    print(f"[data] test:  {test_path}")

    tokenizer = load_tokenizer(args)
    model = load_model(args)
    model_type = getattr(model.config, "model_type", "")
    add_global_attention = model_type == "longformer"
    print(f"[model] type: {model_type}")
    print(f"[model] add_longformer_global_attention: {add_global_attention}")

    train_dataset = AuditDataset(
        load_split(train_path),
        tokenizer,
        args.max_length,
        args.text_field,
        args.label_field,
        add_global_attention,
    )
    val_dataset = AuditDataset(
        load_split(val_path),
        tokenizer,
        args.max_length,
        args.text_field,
        args.label_field,
        add_global_attention,
    )
    test_dataset = (
        AuditDataset(
            load_split(test_path),
            tokenizer,
            args.max_length,
            args.text_field,
            args.label_field,
            add_global_attention,
        )
        if test_path is not None
        else None
    )

    print(f"[data] train size: {len(train_dataset)}")
    print(f"[data] val size:   {len(val_dataset)}")
    if test_dataset is not None:
        print(f"[data] test size:  {len(test_dataset)}")

    collator = AuditDataCollator(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=make_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_flat_metrics,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    val_pred = trainer.predict(val_dataset)
    val_pred_ids = val_pred.predictions.argmax(axis=-1).tolist()
    val_label_ids = val_pred.label_ids.tolist()
    val_metrics = detailed_metrics(val_pred_ids, val_label_ids)
    write_predictions(output_dir / "val_predictions.jsonl", val_dataset, val_pred_ids)

    test_metrics = None
    if test_dataset is not None:
        test_pred = trainer.predict(test_dataset)
        test_pred_ids = test_pred.predictions.argmax(axis=-1).tolist()
        test_label_ids = test_pred.label_ids.tolist()
        test_metrics = detailed_metrics(test_pred_ids, test_label_ids)
        write_predictions(output_dir / "test_predictions.jsonl", test_dataset, test_pred_ids)

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "params": {
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    metrics_path = output_dir / "audit_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved: {metrics_path}")


if __name__ == "__main__":
    main()
