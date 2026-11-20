"""Stage 2: structured Kernel SHAP within dia, exporting data only.

The grouping, masking, KernelExplainer call, and chemical-family aggregation
are implemented in this file so the attribution procedure is directly readable.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import shap

from model import load_predictor
from utils import (
    ROOT, TASK_FILES, DIA_GENOME, read_inputs, selected_tasks, require_new_output,
    paper_tables, sections, field_value, normalize_attribution_sequence,
)

Predictor = Callable[[list[str]], Sequence[float]]
FAMILY_RE = re.compile(r"^\[:\d+([^\]]+)\]$")


def _predict(predictor: Predictor, sequences: list[str]) -> np.ndarray:
    values = np.asarray(predictor(sequences), dtype=np.float64)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.shape != (len(sequences),):
        raise ValueError("Predictor must return one scalar value per input MOFseq.")
    if not np.isfinite(values).all():
        raise ValueError("Predictor returned nonfinite values.")
    return values


@dataclass(frozen=True)
class Feature:
    feature_name: str
    feature_type: str
    marker: str
    positions: tuple[int, ...]
    gene_value: str
    token_role: str = "gene"


def classify_linker_token(token: str) -> str:
    if token == "[pop]" or "Branch" in token or "Ring" in token:
        return "syntax_token"
    if token.startswith("[:"):
        return "group_selfies_fragment"
    if token.startswith("[") and token.endswith("]"):
        return "atom_or_bond_token"
    return "other_token"


def feature_groups(sequence: str) -> tuple[list[str], list[Feature]]:
    """Build the manuscript's stage-2 joint and hybrid attribution units."""
    tokens = normalize_attribution_sequence(sequence).split()
    fields = sections(sequence)
    by_marker = dict(fields)
    if "<T>" not in by_marker or "<cat>" not in by_marker:
        raise ValueError("Joint attribution requires <T> and <cat> fields.")
    topology = " ".join(tokens[i] for i in by_marker["<T>"])
    cat = " ".join(tokens[i] for i in by_marker["<cat>"])
    groups = [Feature(
        f"Topology+Catenation={topology} | cat={cat}",
        "topology_catenation", "<T>+<cat>",
        by_marker["<T>"] + by_marker["<cat>"],
        f"Topology={topology} | Catenation={cat}",
    )]
    for marker, positions in fields:
        if marker in {"<T>", "<cat>"}:
            continue
        tag = marker[1:-1]
        value = " ".join(tokens[i] for i in positions)
        if re.fullmatch(r"<N\d+>", marker):
            groups.append(Feature(f"{tag}={value}", "node", marker, positions, value))
        elif re.fullmatch(r"<I\d+>", marker):
            groups.append(Feature(f"{tag}={value}", "ion", marker, positions, value))
        elif re.fullmatch(r"<L\d+>", marker):
            for local_index, pos in enumerate(positions):
                token = tokens[pos]
                groups.append(Feature(
                    f"{tag}[{local_index:03d}]={token}", "linker_token", marker,
                    (pos,), token, classify_linker_token(token),
                ))
    return tokens, groups


class CoalitionPredictor:
    """Map binary coalitions to fixed-length masked MOFseq inputs."""

    def __init__(self, sequence: str, predictor: Predictor, unk_token: str = "<unk>"):
        self.tokens, self.groups = feature_groups(sequence)
        self.predictor = predictor
        self.unk_token = unk_token
        self.cache: dict[tuple[int, ...], float] = {}

    def masked_sequence(self, coalition: Sequence[float]) -> str:
        row = np.asarray(coalition, dtype=np.float64)
        if row.shape != (len(self.groups),) or not np.isfinite(row).all():
            raise ValueError("Coalition length must match the feature count.")
        if not np.isin(row, [0.0, 1.0]).all():
            raise ValueError("Coalitions must contain only 0 and 1.")
        tokens = self.tokens.copy()
        for active, group in zip(row, self.groups):
            if active:
                continue
            replacement = "0" if group.feature_type == "topology_catenation" else self.unk_token
            for pos in group.positions:
                tokens[pos] = replacement
        return " ".join(tokens)

    def __call__(self, coalitions: np.ndarray) -> np.ndarray:
        matrix = np.asarray(coalitions, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.groups):
            raise ValueError("Coalition matrix must have one column per feature.")
        # Deduplicate coalition evaluations within a call as well as between calls.
        keys = [tuple(int(v) for v in row) for row in matrix]
        for row in matrix:
            self.masked_sequence(row)  # validate before forming integer keys
        missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
        if missing:
            sequences = [self.masked_sequence(key) for key in missing]
            predictions = _predict(self.predictor, sequences)
            self.cache.update((key, float(value)) for key, value in zip(missing, predictions))
        return np.asarray([self.cache[key] for key in keys], dtype=np.float64)


@dataclass
class LocalAttribution:
    sample_id: str
    sequence: str
    features: list[Feature]
    shap_values: np.ndarray
    prediction: float
    baseline_prediction: float
    expected_value: float

    @property
    def additivity_error(self) -> float:
        return float(self.expected_value + self.shap_values.sum() - self.prediction)

    def records(self) -> list[dict]:
        return [
            {
                "sample_id": self.sample_id, "feature_index": i,
                "feature_name": feature.feature_name,
                "feature_type": feature.feature_type, "marker": feature.marker,
                "gene_value": feature.gene_value, "token_role": feature.token_role,
                "shap_value": float(value), "prediction": self.prediction,
                "base_value": self.expected_value,
            }
            for i, (feature, value) in enumerate(zip(self.features, self.shap_values))
        ]

    def summary(self) -> dict:
        return {
            "sample_id": self.sample_id, "n_features": len(self.features),
            "prediction": self.prediction, "baseline_prediction": self.baseline_prediction,
            "expected_value": self.expected_value,
            "additivity_error": self.additivity_error,
        }


def _kernel_one(sequence: str, predictor: Predictor, sample_id: str,
                nsamples: int | str, unk_token: str) -> LocalAttribution:
    if nsamples != "auto" and (not isinstance(nsamples, int) or nsamples <= 0):
        raise ValueError("nsamples must be 'auto' or a positive integer.")
    masked = CoalitionPredictor(sequence, predictor, unk_token)
    count = len(masked.groups)
    background = np.zeros((1, count), dtype=np.float64)
    foreground = np.ones((1, count), dtype=np.float64)
    explainer = shap.KernelExplainer(masked, background,
                                   feature_names=[f.feature_name for f in masked.groups],
                                   link="identity")
    raw = explainer.shap_values(foreground, nsamples=nsamples, l1_reg=0.0,
                               silent=True, gc_collect=True)
    if isinstance(raw, list):
        if len(raw) != 1:
            raise ValueError("Only a single-output property predictor is supported.")
        raw = raw[0]
    values = np.asarray(raw, dtype=np.float64)
    if values.shape not in {(1, count), (1, count, 1), (count,)}:
        raise ValueError(f"Unexpected SHAP value shape: {values.shape}")
    values = values.reshape(count)
    return LocalAttribution(
        sample_id, normalize_attribution_sequence(sequence), masked.groups, values,
        float(masked(foreground)[0]), float(masked(background)[0]),
        float(np.asarray(explainer.expected_value).reshape(-1)[0]),
    )


def kernel_attributions(
    sequences: Sequence[str], predictor: Predictor,
    sample_ids: Sequence[str] | None = None, *, nsamples: int | str = "auto",
    seed: int = 42, unk_token: str = "<unk>",
) -> list[LocalAttribution]:
    """Run stage-2 Kernel SHAP, seeding once for the ordered sample collection.

    The reference, feature grouping, identity link and ``l1_reg=0.0`` follow the
    manuscript protocol. NumPy's previous random state is restored afterwards.
    """
    ids = _sample_ids(sequences, sample_ids)
    previous_state = np.random.get_state()
    np.random.seed(seed)
    try:
        results = []
        for index, (sequence, name) in enumerate(zip(sequences, ids), 1):
            result = _kernel_one(sequence, predictor, name, nsamples, unk_token)
            results.append(result)
            print(f"Kernel SHAP {index}/{len(sequences)}: {name}", flush=True)
        return results
    finally:
        np.random.set_state(previous_state)


def _sample_ids(sequences: Sequence[str], sample_ids: Sequence[str] | None) -> list[str]:
    ids = list(sample_ids) if sample_ids is not None else [f"sample_{i + 1:06d}" for i in range(len(sequences))]
    if len(ids) != len(sequences) or len(ids) != len(set(ids)):
        raise ValueError("Provide one unique sample ID for each sequence.")
    return ids


def fragment_family(token: str, token_role: str | None = None) -> str | None:
    """Post-hoc family assignment; model inputs are never rewritten."""
    role = token_role or classify_linker_token(token)
    if role == "group_selfies_fragment":
        match = FAMILY_RE.fullmatch(token)
        return match.group(1) if match else token
    if token in {"[F]", "[Cl]", "[Br]", "[I]"}:
        return token[1:-1]
    return None


def _aggregate(records: Iterable[dict], family_only: bool) -> tuple[list[dict], list[dict]]:
    # Missing families produce no row: they do not acquire an artificial zero.
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in records:
        kind = row["feature_type"]
        if family_only:
            if kind != "linker_token":
                continue
            value = fragment_family(str(row["gene_value"]), row.get("token_role"))
            if value is None:
                continue
            kind = "linker_family"
        else:
            if kind == "linker_token":
                continue
            value = str(row["gene_value"])
        key = (str(row.get("target", "")), str(row["sample_id"]), kind, value)
        grouped[key].append(row)
    per_sample = []
    for (target, sample_id, kind, value), rows in sorted(grouped.items()):
        phis = np.asarray([r["shap_value"] for r in rows], dtype=float)
        if not np.isfinite(phis).all():
            raise ValueError("Attribution records contain nonfinite values.")
        per_sample.append({
            "target": target, "sample_id": sample_id, "feature_type": kind,
            "feature_key": value, "shap_value": float(phis.sum()),
            "occurrence_count": len(rows), "prediction": float(rows[0]["prediction"]),
        })
    summaries: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in per_sample:
        summaries[(row["target"], row["feature_type"], row["feature_key"])].append(row)
    summary = []
    for (target, kind, value), rows in summaries.items():
        phis = np.asarray([r["shap_value"] for r in rows], dtype=float)
        summary.append({
            "target": target, "feature_type": kind, "feature_key": value,
            "sample_count": len(rows), "mean_shap": float(phis.mean()),
            "median_shap": float(np.median(phis)),
            "mean_abs_shap": float(np.abs(phis).mean()),
            "positive_fraction": float((phis > 0).mean()),
            "occurrence_count": sum(r["occurrence_count"] for r in rows),
        })
    summary.sort(key=lambda row: (row["target"], row["feature_type"], -row["mean_shap"], row["feature_key"]))
    return per_sample, summary


def aggregate_families(records: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """Sum positional variants/copies per MOF, then average present MOFs equally."""
    return _aggregate(records, family_only=True)


def aggregate_features(records: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """Aggregate complete joint topology/interpenetration, node and ion fields."""
    return _aggregate(records, family_only=False)


FUNCTIONAL_GROUP_FAMILIES = {
    "amino", "hydroxy", "nitro", "cyano", "methyl", "methoxy", "sulfanyl",
    "methylsulfanyl", "sulfonic_acid", "sulfonate", "phosphonic_acid",
    "phosphonate_dianion", "F", "Cl", "Br", "I",
}


def dia_paper_points(local_rows):
    features, _ = aggregate_features(local_rows)
    families, _ = aggregate_families(local_rows)
    rows = []
    for row in features:
        kind = row["feature_type"]
        if kind == "topology_catenation":
            cat = int(re.search(r"Catenation=(\d+)", row["feature_key"]).group(1))
            group, key = "topology_catenation", "dia" if cat == 1 else f"{cat}-fold dia"
        elif kind == "node":
            group, key = "metal_nodes", row["feature_key"]
        else:
            continue
        rows.append({**row, "feature_group": group, "feature_key": key})
    for row in families:
        key = row["feature_key"]
        if key in {"carboxylic_acid", "carboxylate"}:
            continue
        group = "functional_groups" if key in FUNCTIONAL_GROUP_FAMILIES else "linker_families"
        rows.append({**row, "feature_group": group})
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/dia_samples.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/recomputed/SHAP_stage2")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--task", choices=["both", *TASK_FILES], default="both")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        frame = read_inputs(args.input)
        if "NAME" not in frame:
            raise ValueError("Stage 2 requires a NAME column.")
        sequences, ids = frame["MOFseq"].tolist(), frame["NAME"].tolist()
        if len(set(sequences)) != len(frame) or len(set(ids)) != len(frame):
            raise ValueError("Stage 2 needs unique MOFseq and NAME values.")
        if any(field_value(sequence, "<T>") != DIA_GENOME for sequence in sequences):
            raise ValueError("Only the standard dia genome is accepted; dia-f is excluded.")
        destination = require_new_output(args.output_dir, args.overwrite)
        for task in selected_tasks(args.task):
            print(f"Stage 2: {task}, {len(frame)} samples; Kernel SHAP auto, l1_reg=0", flush=True)
            predictor = load_predictor(task, args.models_dir, args.device)
            predict = lambda seqs: predictor.predict(seqs, batch_size=args.batch_size, validate=False)

            # Explain each MOF using its own structured binary-coalition game.
            results = kernel_attributions(sequences, predict, ids, nsamples="auto", seed=args.seed)
            error = max(abs(result.additivity_error) for result in results)
            print(f"Maximum local additivity residual: {error:.3e}", flush=True)
            if error > 1e-8:
                raise ValueError("Kernel SHAP failed its additivity check.")

            # Sum repeated/attachment-site variants within each MOF first.
            local_rows = [dict(row, target=task) for result in results for row in result.records()]
            points, summary = paper_tables(dia_paper_points(local_rows), minimum=3, top_n=20)
            target_dir = destination / task
            target_dir.mkdir(exist_ok=True)
            points.to_csv(target_dir / "shap_values.csv", index=False)
            summary.to_csv(target_dir / "summary.csv", index=False)
            print(f"Saved SHAP values and feature summaries (n >= 3): {target_dir}", flush=True)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
