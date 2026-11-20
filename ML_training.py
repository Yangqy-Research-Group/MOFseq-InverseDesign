"""Optional single-target training for the released full-sequence BiGRU.

By default, use the bundled train/validation/test CSV.gz files. New checkpoints
are saved in a separate run folder; pretrained release weights are never replaced.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
import warnings
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from model import FullBiGRURegressor, ModelConfig
from utils import collate_sequences, parse_mofseq, read_vocab, validate_parsed_mofseq


TARGET_COLUMNS = {
    "selectivity": "Selectivity_CH4_N2",
    "ch4_uptake": "CH4_mol_per_kg",
}
_FIELD_TAG = re.compile(r"(?:<T>|<cat>|<N\d+>|<L\d+>|<I\d+>)")


@dataclass
class TrainingConfig:
    """Optimization defaults used for the two final single-target models."""

    seed: int = 521
    batch_size: int = 128
    effective_batch_size: int = 256
    epochs: int = 120
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    patience: int = 18
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_learning_rate: float = 1e-6
    gradient_clip: float = 1.0
    huber_delta: float = 1.0
    token_dropout: float = 0.01
    use_amp: bool = True
    deterministic: bool = True
    num_workers: int = 0

    def validate(self) -> None:
        for name in ("batch_size", "effective_batch_size", "epochs", "patience"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.learning_rate <= 0 or self.huber_delta <= 0:
            raise ValueError("learning_rate and huber_delta must be positive")
        if self.weight_decay < 0 or self.gradient_clip < 0:
            raise ValueError("weight_decay and gradient_clip cannot be negative")
        if not 0 <= self.token_dropout < 1:
            raise ValueError("token_dropout must be in [0, 1)")
        if not 0 < self.scheduler_factor < 1:
            raise ValueError("scheduler_factor must be in (0, 1)")
        if self.scheduler_patience < 0 or self.num_workers < 0:
            raise ValueError("scheduler_patience and num_workers cannot be negative")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch without claiming cross-device bit parity."""
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def transform_targets(values: Any, epsilon: float = 1e-8) -> np.ndarray:
    """Natural logarithm of positive labels, matching the archived float32 path."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Training targets must be finite and strictly positive")
    return np.log(values + epsilon).astype(np.float32)


def target_statistics(values: Any, epsilon: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Estimate normalization statistics from training labels only (ddof=0)."""
    transformed = transform_targets(values, epsilon).reshape(-1, 1)
    mean = transformed.mean(axis=0).astype(np.float32)
    std = np.maximum(transformed.std(axis=0), 1e-6).astype(np.float32)
    return mean, std


def inverse_targets(values: Any, epsilon: float = 1e-8) -> np.ndarray:
    return np.maximum(np.exp(np.asarray(values, dtype=np.float64)) - epsilon, 0.0)


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Metrics on the physical target scale; tied ranks are averaged for SRCC."""
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if true.shape != pred.shape or not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Evaluation arrays must have the same shape and finite values")
    if len(true) == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    residual = true - pred
    total = float(np.sum((true - true.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / (total + 1e-12)) if len(true) > 1 else float("nan")
    true_rank = pd.Series(true).rank().to_numpy()
    pred_rank = pd.Series(pred).rank().to_numpy()
    srcc = float(np.corrcoef(true_rank, pred_rank)[0, 1]) if len(true) > 1 and np.std(true_rank) > 0 and np.std(pred_rank) > 0 else float("nan")
    return {
        "R2": r2,
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "SRCC": srcc,
    }


def _read_split(path: str | Path, target_column: str, split_name: str,
                max_records: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{split_name} CSV not found: {path}")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = ["MOFseq", target_column]
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"{split_name} CSV missing columns: {missing}")
    selected = required + [c for c in ("NAME", "is_experment", "is_experimental") if c in header]
    frame = pd.read_csv(path, usecols=selected, low_memory=False, nrows=max_records)
    if frame.empty or frame[required].isna().any().any():
        raise ValueError(f"{split_name} split is empty or contains missing inputs/targets")
    frame["MOFseq"] = frame["MOFseq"].astype(str).str.strip()
    if frame["MOFseq"].eq("").any():
        raise ValueError(f"{split_name} split contains empty MOFseq inputs")
    frame[target_column] = pd.to_numeric(frame[target_column], errors="raise")
    transform_targets(frame[target_column].to_numpy(dtype=np.float64))
    if "NAME" not in frame:
        frame["NAME"] = [f"{split_name}_{i}" for i in range(len(frame))]
    if frame["NAME"].isna().any():
        raise ValueError(f"{split_name} split contains missing structure names")
    return frame.reset_index(drop=True)


def split_overlap_summary(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Report representation overlap; fixed historical splits are not rewritten."""
    result: dict[str, Any] = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left_seqs = set(frames[left]["MOFseq"])
        right_seqs = set(frames[right]["MOFseq"])
        common = left_seqs & right_seqs
        result[f"{left}_{right}"] = {
            "shared_unique_MOFseq": len(common),
            f"records_in_{right}": int(frames[right]["MOFseq"].isin(common).sum()),
        }
    return result


class _RegressionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, target_column: str, mean: np.ndarray, std: np.ndarray, cfg: ModelConfig):
        self.names = frame["NAME"].astype(str).tolist()
        self.sequences = frame["MOFseq"].tolist()
        self.labels = frame[target_column].to_numpy(dtype=np.float32).reshape(-1, 1)
        self.transformed = transform_targets(self.labels, cfg.transform_epsilon)
        self.targets = ((self.transformed - mean) / std).astype(np.float32)
        self.parsed = []
        for name, sequence in zip(self.names, self.sequences):
            if len(sequence.split()) > cfg.max_full_len:
                raise ValueError(f"{name}: MOFseq exceeds max_full_len={cfg.max_full_len}; refusing truncation")
            parsed = parse_mofseq(sequence)
            try:
                validate_parsed_mofseq(parsed, cfg)
            except ValueError as exc:
                raise ValueError(f"{name}: {exc}") from exc
            self.parsed.append(parsed)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"name": self.names[index], "seq": self.sequences[index], "parsed": self.parsed[index], "target": self.targets[index], "target_raw": self.labels[index]}


class _TrainingCollator:
    def __init__(self, vocabulary: dict[str, int], model_config: ModelConfig, token_dropout: float = 0.0):
        self.vocabulary, self.cfg, self.token_dropout = vocabulary, model_config, token_dropout

    def _consume_archived_component_dropout(self, rows: Sequence[dict[str, Any]]) -> None:
        """Retain the archived collator's NumPy RNG consumption before full input.

        The earlier multi-architecture collator also visited unused component
        tensors. No such tensors are built here, but their dropout draws are
        retained so simplifying the batching does not silently shift this RNG.
        """
        if self.token_dropout <= 0:
            return
        for row in rows:
            for token in row["parsed"]["topo"][:512]:
                if not _FIELD_TAG.fullmatch(token):
                    np.random.random()
        for field, slots, cap in (("nodes", 4, 24), ("linkers", 4, 448), ("ions", 2, 32)):
            for row in rows:
                for tokens in row["parsed"][field][:slots]:
                    for token in tokens[:cap]:
                        if not _FIELD_TAG.fullmatch(token):
                            np.random.random()

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        self._consume_archived_component_dropout(rows)
        batch = collate_sequences(
            [row["seq"] for row in rows], self.vocabulary,
            max_length=self.cfg.max_full_len, token_dropout=self.token_dropout,
            training=self.token_dropout > 0, allow_truncation=False,
        )
        batch.update({
            "target": torch.tensor(np.asarray([row["target"] for row in rows]), dtype=torch.float32),
            "target_raw": torch.tensor(np.asarray([row["target_raw"] for row in rows]), dtype=torch.float32),
            "name": [row["name"] for row in rows], "seq": [row["seq"] for row in rows],
        })
        return batch


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def _evaluate(model, loader, device, mean, std, epsilon, delta):
    model.eval()
    names, sequences, true_parts, predicted_parts = [], [], [], []
    loss_total = 0.0
    count = 0
    for batch in loader:
        batch = _to_device(batch, device)
        prediction = model(batch)
        loss = torch.nn.functional.huber_loss(prediction, batch["target"], delta=delta)
        if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
            raise ValueError("Evaluation produced non-finite predictions/loss")
        loss_total += float(loss.item()) * len(prediction)
        count += len(prediction)
        standardized = prediction.cpu().numpy()
        predicted_parts.append(inverse_targets(standardized * std.reshape(1, -1) + mean.reshape(1, -1), epsilon))
        true_parts.append(batch["target_raw"].cpu().numpy())
        names.extend(batch["name"])
        sequences.extend(batch["seq"])
    true, predicted = np.concatenate(true_parts).reshape(-1), np.concatenate(predicted_parts).reshape(-1)
    metrics = regression_metrics(true, predicted)
    metrics["loss"] = loss_total / count
    return metrics, names, sequences, true, predicted


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def train_model(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
    vocab_csv: str | Path,
    output_dir: str | Path,
    *,
    task: str = "selectivity",
    training_config: TrainingConfig | None = None,
    model_config: ModelConfig | None = None,
    device_name: str = "auto",
    verbose: bool = True,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """Train one task and export a best checkpoint, predictions, and metrics.

    A non-empty output directory is rejected. The supplied pretrained files are
    never updated automatically. The full research splits are required to
    reproduce the reported evaluation scope; a tiny example is only a smoke test.
    """
    if task not in TARGET_COLUMNS:
        raise ValueError(f"task must be one of {tuple(TARGET_COLUMNS)}")
    cfg, network_cfg = training_config or TrainingConfig(), model_config or ModelConfig()
    if smoke_test:
        cfg = replace(cfg, epochs=1)
    cfg.validate()
    if network_cfg.target_transform != "log":
        raise ValueError("The released training workflow supports the log target transform only")
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {out}")
    column = TARGET_COLUMNS[task]
    limits = {"train": 512, "validation": 128, "test": 128} if smoke_test else {}
    if verbose:
        print(f"Loading {task} data{' (SMOKE TEST ONLY)' if smoke_test else ''}...", flush=True)
    frames = {name: _read_split(path, column, name, limits.get(name)) for name, path in (("train", train_csv), ("validation", val_csv), ("test", test_csv))}
    if verbose:
        print("Split records: " + ", ".join(f"{name}={len(frame):,}" for name, frame in frames.items()), flush=True)
    overlap = split_overlap_summary(frames)
    if any(item["shared_unique_MOFseq"] for item in overlap.values()):
        warnings.warn("Identical MOFseq representations occur across splits; see split_overlap in config.json. Splits have not been altered.", UserWarning, stacklevel=2)
    vocabulary = read_vocab(vocab_csv)
    mean, std = target_statistics(frames["train"][column].to_numpy(dtype=np.float64), network_cfg.transform_epsilon)
    datasets = {name: _RegressionDataset(frame, column, mean, std, network_cfg) for name, frame in frames.items()}
    set_seed(cfg.seed, cfg.deterministic)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    common = {"batch_size": cfg.batch_size, "num_workers": cfg.num_workers, "pin_memory": device.type == "cuda", "worker_init_fn": _seed_worker}
    train_loader = DataLoader(datasets["train"], shuffle=True, generator=torch.Generator().manual_seed(cfg.seed), collate_fn=_TrainingCollator(vocabulary, network_cfg, cfg.token_dropout), **common)
    eval_loaders = {name: DataLoader(dataset, shuffle=False, collate_fn=_TrainingCollator(vocabulary, network_cfg), **common) for name, dataset in datasets.items()}
    model = FullBiGRURegressor(network_cfg, vocab_size=max(vocabulary.values()) + 1, pad_id=vocabulary["<pad>"], output_dim=1).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Device: {device}; parameters: {parameter_count:,}; epochs: {cfg.epochs}; output: {out}", flush=True)
        if smoke_test:
            print("Smoke test uses the original architecture and a small data subset; it does not reproduce paper accuracy.", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience, min_lr=cfg.min_learning_rate)
    amp_enabled = cfg.use_amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    accumulation_steps = max(1, math.ceil(cfg.effective_batch_size / cfg.batch_size))
    out.mkdir(parents=True, exist_ok=True)
    settings = {"task": task, "target_column": column, "model_name": "full_bigru", "model": asdict(network_cfg), "training": asdict(cfg), "smoke_test": smoke_test, "split_records": {name: len(frame) for name, frame in frames.items()}, "split_csvs": {"train": str(train_csv), "validation": str(val_csv), "test": str(test_csv)}, "split_overlap": overlap}
    (out / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    best_loss, best_epoch, patience_count = float("inf"), 0, 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss, seen = 0.0, 0
        for step, batch in enumerate(train_loader, start=1):
            batch = _to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                prediction = model(batch)
                loss = torch.nn.functional.huber_loss(prediction, batch["target"], delta=cfg.huber_delta)
                # Match the archived fixed-denominator accumulation convention,
                # including its final partial window.
                scaled_loss = loss / accumulation_steps
            if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
                raise ValueError("Training produced non-finite predictions/loss")
            scaler.scale(scaled_loss).backward()
            if step % accumulation_steps == 0 or step == len(train_loader):
                if cfg.gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.item()) * len(prediction)
            seen += len(prediction)
            if verbose and (step == 1 or step % 100 == 0 or step == len(train_loader)):
                print(f"Epoch {epoch:03d}, batch {step}/{len(train_loader)}: loss={loss.item():.6f}", flush=True)
        validation = _evaluate(model, eval_loaders["validation"], device, mean, std, network_cfg.transform_epsilon, cfg.huber_delta)[0]
        val_loss = validation["loss"]
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "train_loss": running_loss / seen, **{f"validation_{k}": v for k, v in validation.items()}})
        if verbose:
            print(f"Epoch {epoch:03d}: train={running_loss / seen:.6f}, validation={val_loss:.6f}, R2={validation['R2']:.4f}", flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch, patience_count = val_loss, epoch, 0
            checkpoint = {"schema_version": 2, "model_name": "full_bigru", "model_state_dict": model.state_dict(), "config": {**asdict(network_cfg), "model_name": "full_bigru", "seed": cfg.seed}, "target_cols": [column], "target_transform": "log", "target_mean": mean.tolist(), "target_std": std.tolist(), "vocab_size": max(vocabulary.values()) + 1, "pad_id": vocabulary["<pad>"], "unk_id": vocabulary["<unk>"], "token_to_id": dict(vocabulary), "output_dim": 1, "best_epoch": epoch, "best_val_loss": best_loss, "parameter_count": sum(p.numel() for p in model.parameters())}
            torch.save(checkpoint, out / "best_model.pt")
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                break
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    checkpoint = torch.load(out / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    results = {}
    rows = []
    for split, loader in eval_loaders.items():
        metrics, names, sequences, true, predicted = _evaluate(model, loader, device, mean, std, network_cfg.transform_epsilon, cfg.huber_delta)
        results[split] = metrics
        rows.append({"task": task, "split": split, "n": len(true), **metrics})
        pd.DataFrame({"NAME": names, "MOFseq": sequences, f"true_{column}": true, f"pred_{column}": predicted, f"error_{column}": predicted - true}).to_csv(out / f"{split}_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    summary = {"task": task, "seed": cfg.seed, "best_epoch": best_epoch, "best_validation_loss": best_loss, "elapsed_seconds": time.perf_counter() - start, "parameter_count": sum(p.numel() for p in model.parameters()), "smoke_test": smoke_test, "metrics": results}
    # Keep run metadata in one file rather than exporting duplicate summaries.
    settings.update({key: value for key, value in summary.items() if key != "metrics"})
    (out / "config.json").write_text(json.dumps(_json_safe(settings), indent=2, allow_nan=False), encoding="utf-8")
    return summary


# Default resources are resolved relative to this script, not the current folder.
ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train one MOFseq property model using fixed historical splits."
    )
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--val-csv", type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--vocab", type=Path, default=ROOT / "data" / "vocab.csv")
    parser.add_argument("--task", choices=list(TARGET_COLUMNS), default="selectivity")
    parser.add_argument("--output-dir", type=Path,
                        help="New output folder; default: results/training/<task>_<unique-run-id>.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--effective-batch-size", type=int, default=256)
    parser.add_argument("--smoke-test", action="store_true",
                        help="One epoch with the original architecture on 512/128/128 real records; not full retraining.")
    args = parser.parse_args(argv)

    explicit = (args.train_csv, args.val_csv, args.test_csv)
    if any(path is not None for path in explicit) and not all(path is not None for path in explicit):
        parser.error("Supply all three of --train-csv, --val-csv and --test-csv, or none to use bundled data.")
    if not any(path is not None for path in explicit):
        explicit = tuple(ROOT / "data" / filename for filename in ("train.csv.gz", "val.csv.gz", "test.csv.gz"))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
    prefix = f"smoke_{args.task}" if args.smoke_test else args.task
    output = args.output_dir or ROOT / "results" / "training" / f"{prefix}_{run_id}"
    training = TrainingConfig(
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        effective_batch_size=args.effective_batch_size,
    )
    try:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise FileExistsError(f"Refusing to overwrite existing output: {output}")
        summary = train_model(
            *explicit, args.vocab, output, task=args.task,
            training_config=training, device_name=args.device,
            smoke_test=args.smoke_test,
        )
        print(f"Saved the best model, predictions and metrics to {output}", flush=True)
        print(f"Best epoch: {summary['best_epoch']}", flush=True)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(2, f"ML_training.py: {exc}\n")


if __name__ == "__main__":
    main()

