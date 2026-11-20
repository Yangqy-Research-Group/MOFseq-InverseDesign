"""Neural network architecture, checkpoint loading, and property prediction.

The module/layer names match the original pretrained checkpoints exactly.
"""
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence
import pickle

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import (
    ROOT, TASK_FILES, TARGET_COLUMNS, choose_device, collate_sequences,
    normalize_mofseq, read_vocab, split_tokens, validate_vocab,
)

@dataclass
class ModelConfig:
    """Architecture and preprocessing settings needed by the final BiGRU."""

    model_name: str = "full_bigru"
    full_dim: int = 384
    full_gru_hidden_dim: int = 288
    full_gru_layers: int = 3
    head_dim: int = 512
    dropout: float = 0.15
    max_full_len: int = 640
    max_cat: int = 8
    target_transform: str = "log"
    transform_epsilon: float = 1e-8

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ModelConfig":
        """Read relevant settings while ignoring legacy experimental options."""
        names = {field.name for field in fields(cls)}
        config = cls(**{key: value for key, value in values.items() if key in names})
        config.validate()
        return config

    def validate(self) -> None:
        if self.model_name != "full_bigru":
            raise ValueError("This package supports only the final full_bigru architecture.")
        for name in ("full_dim", "full_gru_hidden_dim", "full_gru_layers", "max_full_len", "max_cat"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.head_dim < 2:
            raise ValueError("head_dim must be at least 2.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if self.target_transform not in {"log", "log1p", "none"}:
            raise ValueError("target_transform must be log, log1p, or none.")
        if self.transform_epsilon < 0:
            raise ValueError("transform_epsilon cannot be negative.")


class MaskedAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        empty = ~mask.any(dim=1)
        score = self.attention(x).squeeze(-1).masked_fill(~mask, -1e4)
        if empty.any():
            score = score.clone()
            score[empty] = 0.0
        weight = torch.softmax(score, dim=1).masked_fill(~mask, 0.0)
        pooled = torch.bmm(weight.unsqueeze(1), x).squeeze(1)
        if empty.any():
            pooled = pooled.clone()
            pooled[empty] = 0.0
        return pooled


class GRUTokenEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        embedding_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        out_dim: int,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.pool = MaskedAttentionPooling(hidden_dim * 2, dropout)
        self.projection = nn.Linear(hidden_dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1)
        empty = lengths == 0
        safe_lengths = lengths.clamp(min=1).detach().cpu()
        x = self.embedding(ids)
        packed = pack_padded_sequence(x, safe_lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=ids.shape[1])
        pooled = self.pool(out, mask)
        encoded = self.norm(self.projection(pooled))
        if empty.any():
            encoded = encoded.clone()
            encoded[empty] = 0.0
        return encoded


class RegressionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FullBiGRURegressor(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int, pad_id: int, output_dim: int = 1):
        super().__init__()
        cfg.validate()
        self.model_name = "full_bigru"
        self.encoder = GRUTokenEncoder(
            vocab_size, pad_id, cfg.full_dim, cfg.full_gru_hidden_dim,
            cfg.full_gru_layers, cfg.dropout, cfg.full_dim,
        )
        self.head = RegressionHead(cfg.full_dim, cfg.head_dim, output_dim, cfg.dropout)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.encoder(batch["full_seq"], batch["full_seq_mask"]))


def build_model(cfg: ModelConfig, vocab_size: int, pad_id: int, output_dim: int = 1) -> FullBiGRURegressor:
    return FullBiGRURegressor(cfg, vocab_size, pad_id, output_dim)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_checkpoint(path: str | Path, *, trusted_checkpoint: bool = False) -> dict[str, Any]:
    """Load tensor/primitive checkpoints safely by default.

    ``trusted_checkpoint=True`` enables legacy pickle loading for old research
    checkpoints that contain NumPy metadata.  Never enable it for an untrusted
    downloaded checkpoint, since pickle can execute arbitrary Python code.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=not trusted_checkpoint)
    except TypeError as exc:
        raise RuntimeError("A PyTorch version supporting weights_only loading is required.") from exc
    except pickle.UnpicklingError as exc:
        raise ValueError(
            "Checkpoint could not be loaded with weights_only=True. The released "
            "checkpoints use safe tensor/primitive metadata. For a known, trusted "
            "legacy research file only, pass trusted_checkpoint=True."
        ) from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a dictionary.")
    return checkpoint


def config_from_checkpoint(checkpoint: Mapping[str, Any]) -> ModelConfig:
    saved = checkpoint.get("config")
    if not isinstance(saved, Mapping):
        raise ValueError("Checkpoint does not contain a configuration dictionary.")
    values = dict(saved)
    values["model_name"] = checkpoint.get("model_name", values.get("model_name", "full_bigru"))
    return ModelConfig.from_mapping(values)


def inverse_transform_targets_np(values: np.ndarray, cfg: ModelConfig) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if cfg.target_transform == "log":
        result = np.exp(values) - cfg.transform_epsilon
    elif cfg.target_transform == "log1p":
        result = np.expm1(values)
    else:
        result = values
    return np.maximum(result, 0.0).astype(np.float64)


class MOFSeqPredictor:
    """Load one single-target model once and predict physical target values."""

    REQUIRED_FIELDS = {
        "model_state_dict", "config", "target_cols", "target_mean", "target_std",
        "vocab_size", "pad_id", "unk_id", "token_to_id", "output_dim", "model_name",
    }

    def __init__(
        self,
        checkpoint_path: str | Path,
        device_name: str = "auto",
        *,
        vocab_path: str | Path | None = None,
        trusted_checkpoint: bool = False,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = choose_device(device_name)
        self.checkpoint = load_checkpoint(self.checkpoint_path, trusted_checkpoint=trusted_checkpoint)
        missing = sorted(self.REQUIRED_FIELDS.difference(self.checkpoint))
        if missing:
            raise ValueError(f"Checkpoint is missing required fields: {missing}")
        self.cfg = config_from_checkpoint(self.checkpoint)
        self.target_cols = list(self.checkpoint["target_cols"])
        self.output_dim = int(self.checkpoint["output_dim"])
        if self.output_dim != 1 or len(self.target_cols) != 1:
            raise ValueError("The released workflow requires a separate single-output model per target.")
        self.token_to_id = {str(token): int(identifier) for token, identifier in self.checkpoint["token_to_id"].items()}
        validate_vocab(self.token_to_id)
        self.pad_id = int(self.checkpoint["pad_id"])
        self.unk_id = int(self.checkpoint["unk_id"])
        if self.pad_id != self.token_to_id["<pad>"] or self.unk_id != self.token_to_id["<unk>"]:
            raise ValueError("Checkpoint special-token ids disagree with its embedded vocabulary.")
        self.vocab_size = int(self.checkpoint["vocab_size"])
        if self.vocab_size <= max(self.token_to_id.values()):
            raise ValueError("Checkpoint vocabulary contains an id outside its embedding table.")
        if vocab_path is not None and read_vocab(vocab_path) != self.token_to_id:
            raise ValueError("The supplied vocabulary differs from the checkpoint vocabulary.")
        self.target_mean = np.asarray(self.checkpoint["target_mean"], dtype=np.float32).reshape(1, -1)
        self.target_std = np.asarray(self.checkpoint["target_std"], dtype=np.float32).reshape(1, -1)
        if self.target_mean.shape != (1, 1) or self.target_std.shape != (1, 1):
            raise ValueError("Target normalization statistics do not match the output dimension.")
        if not np.isfinite(self.target_mean).all() or not np.isfinite(self.target_std).all() or (self.target_std <= 0).any():
            raise ValueError("Target normalization statistics must be finite, with positive standard deviation.")
        self.model = build_model(self.cfg, self.vocab_size, self.pad_id, self.output_dim)
        self.model.load_state_dict(self.checkpoint["model_state_dict"], strict=True)
        self.model.to(self.device).eval()

    def normalize_and_validate(self, mofseq: str, index: int = 0) -> str:
        try:
            return normalize_mofseq(mofseq, self.cfg)
        except ValueError as exc:
            raise ValueError(f"MOFseq at index {index}: {exc}") from exc

    def unknown_tokens(self, mofseq: str) -> list[str]:
        return sorted({token for token in split_tokens(mofseq) if token not in self.token_to_id})

    @torch.inference_mode()
    def predict(
        self,
        mofseqs: Sequence[str],
        batch_size: int = 128,
        *,
        validate: bool = True,
        allow_truncation: bool = False,
    ) -> np.ndarray:
        """Predict an ``(n, 1)`` array after inverse log standardization.

        ``validate=False`` is reserved for explicitly constructed attribution
        masks, which need not describe a chemically valid complete structure.
        """
        if isinstance(mofseqs, str):
            raise TypeError("Pass a sequence of MOFseq strings, or use predict_one().")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        sequences = []
        for index, sequence in enumerate(mofseqs):
            try:
                sequences.append(normalize_mofseq(sequence, self.cfg, validate=validate))
            except ValueError as exc:
                raise ValueError(f"MOFseq at index {index}: {exc}") from exc
        if not sequences:
            return np.empty((0, self.output_dim), dtype=np.float64)
        predictions = []
        for start in range(0, len(sequences), batch_size):
            batch = collate_sequences(
                sequences[start:start + batch_size], self.token_to_id,
                self.pad_id, self.unk_id, self.cfg.max_full_len,
                allow_truncation=allow_truncation,
            )
            batch = {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}
            standardized = self.model(batch).detach().cpu().numpy()
            transformed = standardized * self.target_std + self.target_mean
            predictions.append(inverse_transform_targets_np(transformed, self.cfg))
        return np.concatenate(predictions, axis=0)

    def predict_one(self, mofseq: str) -> dict[str, float]:
        values = self.predict([mofseq], batch_size=1)[0]
        return {target: float(value) for target, value in zip(self.target_cols, values)}


def load_predictor(task, models_dir=None, device="auto"):
    directory = ROOT / "models" if models_dir is None else Path(models_dir)
    predictor = MOFSeqPredictor(directory / TASK_FILES[task], device_name=device)
    if predictor.target_cols != [TARGET_COLUMNS[task]]:
        raise ValueError(f"Checkpoint target does not match task: {task}")
    return predictor
