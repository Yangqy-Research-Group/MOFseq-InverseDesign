"""Small shared helpers for paths, MOFseq parsing, batching, and CSV summaries.

No neural-network architecture or Shapley/Kernel SHAP algorithm is defined here.
"""
import csv
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

TAG_PATTERN = re.compile(r"(<T>|<cat>|<N\d+>|<L\d+>|<I\d+>)")


NODE_TAG = re.compile(r"<N\d+>")


LINKER_TAG = re.compile(r"<L\d+>")


ION_TAG = re.compile(r"<I\d+>")


def split_tokens(text: str) -> list[str]:
    return text.strip().split() if isinstance(text, str) else []


def read_vocab(vocab_csv: str | Path) -> dict[str, int]:
    """Read the first two columns of a token/id CSV without changing its ids."""
    with Path(vocab_csv).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or len(header) < 2:
            raise ValueError("Vocabulary CSV must contain a header and token/id columns.")
        token_to_id: dict[str, int] = {}
        for line_number, row in enumerate(reader, 2):
            if not row or not any(row):
                continue
            if len(row) < 2:
                raise ValueError(f"Vocabulary line {line_number} has fewer than two columns.")
            token = row[0]
            try:
                identifier = int(row[1])
            except ValueError as exc:
                raise ValueError(f"Invalid vocabulary id on line {line_number}.") from exc
            if token in token_to_id:
                raise ValueError(f"Duplicate vocabulary token: {token!r}.")
            token_to_id[token] = identifier
    validate_vocab(token_to_id)
    return token_to_id


def validate_vocab(token_to_id: Mapping[str, int]) -> None:
    if not token_to_id:
        raise ValueError("The vocabulary is empty.")
    ids = list(token_to_id.values())
    if any(not isinstance(value, int) or value < 0 for value in ids):
        raise ValueError("Vocabulary ids must be non-negative integers.")
    if len(ids) != len(set(ids)):
        raise ValueError("Different vocabulary tokens cannot share an id.")
    for special in ("<pad>", "<unk>"):
        if special not in token_to_id:
            raise ValueError(f"Vocabulary is missing {special}.")


def parse_mofseq(seq: str) -> dict[str, Any]:
    """Split topology, interpenetration, node, linker, and ion fields."""
    topology: list[str] = []
    cat = 1
    cat_present = False
    nodes: list[list[str]] = []
    linkers: list[list[str]] = []
    ions: list[list[str]] = []
    current_tag: str | None = None
    current_buf: list[str] = []

    def flush(tag: str | None, buf: list[str]) -> None:
        nonlocal topology, cat, cat_present
        if tag == "<T>":
            topology = list(buf)
        elif tag == "<cat>":
            cat_present = True
            if buf:
                try:
                    cat = int(float(buf[0]))
                except (ValueError, TypeError, OverflowError):
                    cat = 1
        elif tag is not None and NODE_TAG.fullmatch(tag) and buf:
            nodes.append(list(buf))
        elif tag is not None and LINKER_TAG.fullmatch(tag) and buf:
            linkers.append(list(buf))
        elif tag is not None and ION_TAG.fullmatch(tag) and buf:
            ions.append(list(buf))

    for token in split_tokens(seq):
        if TAG_PATTERN.fullmatch(token):
            flush(current_tag, current_buf)
            current_tag, current_buf = token, []
        else:
            current_buf.append(token)
    flush(current_tag, current_buf)
    return {"topo": topology, "cat": cat, "cat_present": cat_present,
            "nodes": nodes, "linkers": linkers, "ions": ions}


def validate_parsed_mofseq(parsed: Mapping[str, Any], cfg: Any) -> None:
    errors = []
    if not parsed["topo"]:
        errors.append("missing topology content")
    if not parsed.get("cat_present", False):
        errors.append("missing <cat> section")
    if not parsed["nodes"]:
        errors.append("missing node section")
    if not parsed["linkers"]:
        errors.append("missing linker section")
    cat = int(parsed["cat"])
    if not 1 <= cat <= cfg.max_cat:
        errors.append(f"interpenetration {cat} outside 1..{cfg.max_cat}")
    if errors:
        raise ValueError("Invalid MOFseq: " + "; ".join(errors))


def normalize_mofseq(mofseq: str, cfg: Any, *, validate: bool = True) -> str:
    if not isinstance(mofseq, str):
        raise ValueError("MOFseq must be a string.")
    normalized = " ".join(mofseq.split())
    if not normalized:
        raise ValueError("MOFseq is empty.")
    if validate:
        validate_parsed_mofseq(parse_mofseq(normalized), cfg)
    return normalized


def collate_sequences(
    sequences: Sequence[str],
    token_to_id: Mapping[str, int],
    pad_id: int | None = None,
    unk_id: int | None = None,
    max_length: int = 640,
    token_dropout: float = 0.0,
    training: bool = False,
    allow_truncation: bool = False,
) -> dict[str, torch.Tensor]:
    """Dynamically right-pad complete sequences using the checkpoint vocabulary.

    Unknown tokens map to ``<unk>``.  Long sequences are rejected by default,
    rather than silently truncating a linker or another structural field.
    """
    if not sequences:
        raise ValueError("Cannot collate an empty batch.")
    if max_length < 1:
        raise ValueError("max_length must be positive.")
    if not 0 <= token_dropout < 1:
        raise ValueError("token_dropout must be in [0, 1).")
    pad_id = token_to_id["<pad>"] if pad_id is None else int(pad_id)
    unk_id = token_to_id["<unk>"] if unk_id is None else int(unk_id)
    token_lists = [split_tokens(sequence) for sequence in sequences]
    for index, tokens in enumerate(token_lists):
        if not tokens:
            raise ValueError(f"MOFseq at batch index {index} is empty.")
        if len(tokens) > max_length and not allow_truncation:
            raise ValueError(
                f"MOFseq at batch index {index} contains {len(tokens)} tokens; "
                f"the model limit is {max_length}."
            )
    length = max(1, min(max_length, max(map(len, token_lists))))
    ids = torch.full((len(sequences), length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), length), dtype=torch.bool)
    for row, tokens in enumerate(token_lists):
        visible = tokens[:length]
        values = [token_to_id.get(token, unk_id) for token in visible]
        if training and token_dropout:
            for column, token in enumerate(visible):
                if not TAG_PATTERN.fullmatch(token) and np.random.random() < token_dropout:
                    values[column] = unk_id
        ids[row, :len(values)] = torch.tensor(values, dtype=torch.long)
        mask[row, :len(values)] = True
    return {"full_seq": ids, "full_seq_mask": mask}


def choose_device(device_name: str = "auto") -> torch.device:
    if device_name.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available to PyTorch.")
    return device


ROOT = Path(__file__).resolve().parent


TASK_FILES = {"selectivity": "selectivity.pt", "ch4_uptake": "ch4_uptake.pt"}


TARGET_COLUMNS = {"selectivity": "Selectivity_CH4_N2", "ch4_uptake": "CH4_mol_per_kg"}


PREDICTION_COLUMNS = {"selectivity": "pred_selectivity", "ch4_uptake": "pred_ch4_uptake_mmol_g"}


def read_inputs(path, limit=None):
    import pandas as pd
    frame = pd.read_csv(path, low_memory=False)
    if "MOFseq" not in frame or frame["MOFseq"].isna().any():
        raise ValueError("Input requires a MOFseq column without missing values.")
    frame["MOFseq"] = frame["MOFseq"].map(lambda text: " ".join(str(text).split()))
    if frame["MOFseq"].eq("").any():
        raise ValueError("Input contains an empty MOFseq.")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive.")
        frame = frame.head(limit).copy()
    if frame.empty:
        raise ValueError("Input has no records.")
    return frame


def selected_tasks(task):
    return list(TASK_FILES) if task == "both" else [task]


TAG_RE = re.compile(r"^<(T|cat|N\d+|L\d+|I\d+)>$")


DIA_GENOME = "3 1 2 0 0 0 1 2 0 0 1 1 2 0 1 0 1 2 1 0 0"


def normalize_attribution_sequence(sequence: str) -> str:
    """Normalize whitespace without changing token identities or their order."""
    if not isinstance(sequence, str):
        raise TypeError("MOFseq must be a string.")
    normalized = " ".join(sequence.split())
    if not normalized:
        raise ValueError("MOFseq must not be empty.")
    return normalized


def sections(sequence: str) -> list[tuple[str, tuple[int, ...]]]:
    """Return field markers and content positions in the normalized sequence."""
    tokens = normalize_attribution_sequence(sequence).split()
    starts = []
    for i, token in enumerate(tokens):
        if TAG_RE.fullmatch(token):
            starts.append(i)
        elif token.startswith("<") and token.endswith(">"):
            raise ValueError(f"Unsupported MOFseq field marker: {token}")
    if not starts or starts[0] != 0:
        raise ValueError("MOFseq must begin with a recognized field marker.")
    markers = [tokens[i] for i in starts]
    if len(markers) != len(set(markers)):
        raise ValueError("MOFseq field markers must be unique.")
    result = []
    for j, start in enumerate(starts):
        stop = starts[j + 1] if j + 1 < len(starts) else len(tokens)
        positions = tuple(range(start + 1, stop))
        if not positions:
            raise ValueError(f"MOFseq field {tokens[start]} has no content.")
        result.append((tokens[start], positions))
    return result


def field_value(sequence: str, marker: str) -> str:
    tokens = normalize_attribution_sequence(sequence).split()
    matches = [positions for name, positions in sections(sequence) if name == marker]
    if len(matches) != 1:
        raise ValueError(f"MOFseq must contain exactly one {marker} field.")
    return " ".join(tokens[i] for i in matches[0])


def require_new_output(directory, overwrite=False):
    directory = Path(directory)
    if directory.exists() and not directory.is_dir():
        raise FileExistsError(f"Not a directory: {directory}")
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError(f"Output is not empty: {directory}. Use --overwrite or a new --output-dir.")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def paper_tables(points, minimum, top_n):
    """Select the manuscript's frequency/rank scope, without token-level exports."""
    columns = ["sample_id", "feature_group", "feature_key", "shap_value", "prediction"]
    points = points[columns].copy()
    keys = ["feature_group", "feature_key"]
    if points.duplicated(["sample_id", *keys]).any():
        raise ValueError("Expected one within-MOF sum per feature before averaging.")
    summary = points.groupby(keys, sort=False).agg(
        n=("sample_id", "nunique"), mean_shap=("shap_value", "mean"),
        median_shap=("shap_value", "median"),
        positive_fraction=("shap_value", lambda values: float((values > 0).mean())),
    ).reset_index()
    summary = summary.loc[summary["n"].ge(minimum)].sort_values(
        ["feature_group", "mean_shap", "n", "feature_key"],
        ascending=[True, False, False, True], kind="stable")
    summary = summary.groupby("feature_group", sort=False).head(top_n).reset_index(drop=True)
    points = points.merge(summary[keys], on=keys, how="inner", validate="many_to_one")
    return points, summary
