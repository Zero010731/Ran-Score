import argparse
import importlib.util
from pathlib import Path


PATCH_TAG = "# RanAudit ALSPC patch"


def package_root():
    spec = importlib.util.find_spec("llamafactory")
    if spec is None or spec.origin is None:
        raise RuntimeError("Cannot locate installed llamafactory package in this Python environment.")
    return Path(spec.origin).resolve().parent


def backup_once(path):
    backup = path.with_suffix(path.suffix + ".ranaudit_alspc_bak")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return backup


def restore_once(path):
    backup = path.with_suffix(path.suffix + ".ranaudit_alspc_bak")
    if not backup.exists():
        print(f"skip restore; missing backup: {backup}")
        return
    path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"restored {path}")


def replace_once(text, old, new, path):
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Patch target not found in {path}: {old[:120]!r}")
    return text.replace(old, new, 1)


def maybe_write(path, text, dry_run):
    if dry_run:
        print(f"dry-run: would update {path}")
        return
    path.write_text(text, encoding="utf-8")


def patch_converter(root, dry_run=False):
    path = root / "data" / "converter.py"
    text = path.read_text(encoding="utf-8")
    if not dry_run:
        backup_once(path)
    old = '''        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": example[self.dataset_attr.system] if self.dataset_attr.system else "",
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        return output
'''
    new = '''        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": example[self.dataset_attr.system] if self.dataset_attr.system else "",
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        if "alspc_margin" in example:
            output["alspc_margin"] = float(example["alspc_margin"])
        return output
'''
    maybe_write(path, replace_once(text, old, new, path), dry_run)
    print(f"patched {path}")


def patch_pairwise(root, dry_run=False):
    path = root / "data" / "processor" / "pairwise.py"
    text = path.read_text(encoding="utf-8")
    if not dry_run:
        backup_once(path)
    old = '''            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs
'''
    new = '''            model_inputs["audios"].append(examples["_audios"][i])
            if "alspc_margin" in examples:
                model_inputs.setdefault("alspc_margin", [])
                model_inputs["alspc_margin"].append(float(examples["alspc_margin"][i]))

        return model_inputs
'''
    maybe_write(path, replace_once(text, old, new, path), dry_run)
    print(f"patched {path}")


def patch_collator(root, dry_run=False):
    path = root / "data" / "collator.py"
    text = path.read_text(encoding="utf-8")
    if not dry_run:
        backup_once(path)
    old = '''        return super().__call__(concatenated_features)
'''
    new = '''        batch = super().__call__(concatenated_features)
        if features and "alspc_margin" in features[0]:
            batch["alspc_margin"] = torch.tensor([feature["alspc_margin"] for feature in features], dtype=torch.float)
        return batch
'''
    maybe_write(path, replace_once(text, old, new, path), dry_run)
    print(f"patched {path}")


def patch_trainer(root, dry_run=False):
    path = root / "train" / "dpo" / "trainer.py"
    text = path.read_text(encoding="utf-8")
    if not dry_run:
        backup_once(path)

    old = '''        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss

        prefix = "eval_" if train_eval == "eval" else ""
'''
    new = '''        sft_loss = -policy_chosen_logps_avg
        alspc_margin = batch.pop("alspc_margin", None)
        if alspc_margin is not None and self.loss_type == "sigmoid":
            alspc_margin = alspc_margin.to(losses.device, dtype=losses.dtype)
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = reference_chosen_logps - reference_rejected_logps
            logits = pi_logratios - ref_logratios - alspc_margin
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss

        prefix = "eval_" if train_eval == "eval" else ""
        if alspc_margin is not None:
            metrics[f"{prefix}alspc/margin"] = alspc_margin.mean().item()
'''
    maybe_write(path, replace_once(text, old, new, path), dry_run)
    print(f"patched {path}")


def restore(root):
    restore_once(root / "data" / "converter.py")
    restore_once(root / "data" / "processor" / "pairwise.py")
    restore_once(root / "data" / "collator.py")
    restore_once(root / "train" / "dpo" / "trainer.py")


def main():
    parser = argparse.ArgumentParser(description="Patch an installed LLaMA-Factory package to consume ALSPC margins.")
    parser.add_argument("--dry-run", action="store_true", help="Check patch targets without modifying files.")
    parser.add_argument("--restore", action="store_true", help="Restore files from .ranaudit_alspc_bak backups.")
    args = parser.parse_args()

    root = package_root()
    print(f"llamafactory root: {root}")
    if args.restore:
        restore(root)
        return

    patch_converter(root, dry_run=args.dry_run)
    patch_pairwise(root, dry_run=args.dry_run)
    patch_collator(root, dry_run=args.dry_run)
    patch_trainer(root, dry_run=args.dry_run)
    if args.dry_run:
        print("ALSPC patch dry-run completed.")
    else:
        print("ALSPC patch applied. Backup files end with .ranaudit_alspc_bak")


if __name__ == "__main__":
    main()
