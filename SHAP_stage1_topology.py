"""Stage 1: exact topology-field Shapley attribution; CSV output only.

The entire <T> field is one binary feature. All other fields remain fixed,
so its exact Shapley value is f(complete) - f(topology masked). There are only
two states to evaluate; no coalition sampler or SHAP package is needed here.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from model import load_predictor
from utils import (
    ROOT,
    TASK_FILES,
    field_value,
    paper_tables,
    read_inputs,
    require_new_output,
    sections,
    selected_tasks,
)


def _normalize_sequence(sequence: str) -> str:
    if not isinstance(sequence, str):
        raise TypeError("MOFseq must be a string.")
    normalized = " ".join(sequence.split())
    if not normalized:
        raise ValueError("MOFseq must not be empty.")
    return normalized


def mask_topology(sequence: str) -> str:
    """Replace topology content with 0, retaining markers and sequence length."""
    tokens = _normalize_sequence(sequence).split()
    matches = [positions for marker, positions in sections(sequence) if marker == "<T>"]
    if len(matches) != 1:
        raise ValueError("Topology attribution requires exactly one <T> field.")
    for position in matches[0]:
        tokens[position] = "0"
    return " ".join(tokens)


def _predict_values(predictor: Callable, sequences: list[str]) -> np.ndarray:
    values = np.asarray(predictor(sequences), dtype=np.float64)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.shape != (len(sequences),) or not np.isfinite(values).all():
        raise ValueError("Predictor must return one finite scalar per MOFseq.")
    return values


def exact_topology_attributions(
    sequences: Sequence[str],
    predictor: Callable,
    sample_ids: Sequence[str] | None = None,
) -> list[dict]:
    """Evaluate both topology states on the original physical property scale.

    The reference retains each MOF's non-topological fields and can therefore
    differ across samples. This is not a multifeature joint explanation.
    """
    normalized = [_normalize_sequence(sequence) for sequence in sequences]
    ids = list(sample_ids) if sample_ids is not None else [
        f"sample_{index + 1:06d}" for index in range(len(normalized))
    ]
    if len(ids) != len(normalized) or len(ids) != len(set(ids)):
        raise ValueError("Provide one unique sample ID per MOFseq.")
    if not normalized:
        return []

    complete = _predict_values(predictor, normalized)
    reference = _predict_values(predictor, [mask_topology(sequence) for sequence in normalized])
    return [
        {
            "sample_id": sample_id,
            "MOFseq": sequence,
            "topology_genome": field_value(sequence, "<T>"),
            "prediction": float(full),
            "baseline_prediction": float(base),
            "topology_shap": float(full - base),
        }
        for sample_id, sequence, full, base in zip(ids, normalized, complete, reference)
    ]


def validated_topology_groups(frame: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Use the MOFseq topology genome as identity; names are display metadata.

    Both directions of the genome/name mapping must be unambiguous. A topology
    name can never silently merge distinct genomes, and a missing historical
    name cannot split inputs that have the same genome.
    """
    required = {"NAME", "MOFseq", "topology_genome", "topology_name"}
    if not required.issubset(frame):
        raise ValueError("Stage 1 input requires NAME, MOFseq, topology_genome, and topology_name.")
    if frame[list(required)].isna().any().any():
        raise ValueError("Stage 1 topology identities and display names must not be missing.")
    genomes = frame["MOFseq"].map(lambda sequence: field_value(sequence, "<T>"))
    supplied = frame["topology_genome"].map(_normalize_sequence)
    mismatch = genomes.ne(supplied)
    if mismatch.any():
        names = frame.loc[mismatch, "NAME"].astype(str).head(5).tolist()
        raise ValueError(f"topology_genome disagrees with the MOFseq <T> field: {names}")
    labels = frame["topology_name"].astype(str).str.strip()
    if labels.eq("").any():
        raise ValueError("Topology display names must not be empty.")
    identities = pd.DataFrame({
        "topology_genome": genomes,
        "topology_name": labels,
        "NAME": frame["NAME"],
    })
    if identities.groupby("topology_genome")["topology_name"].nunique().gt(1).any():
        raise ValueError("One topology genome has multiple display names; resolve the mapping before analysis.")
    if identities.groupby("topology_name")["topology_genome"].nunique().gt(1).any():
        raise ValueError("One display name refers to multiple topology genomes; use distinct labels before analysis.")
    groups = identities.groupby("topology_genome", sort=True).agg(
        topology_name=("topology_name", "first"), n=("NAME", "nunique"),
    ).reset_index()
    if int(groups["n"].sum()) != len(frame):
        raise ValueError("Topology group counts do not partition the unique analysis samples.")
    return genomes, groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/attribution_samples.csv.gz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/recomputed/SHAP_stage1")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--task", choices=["both", *TASK_FILES], default="both")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        frame = read_inputs(args.input)
        genomes, groups = validated_topology_groups(frame)
        if frame["MOFseq"].duplicated().any() or frame["NAME"].duplicated().any():
            raise ValueError("Stage 1 needs unique MOFseq and NAME values.")
        display_names = groups.set_index("topology_genome")["topology_name"]
        destination = require_new_output(args.output_dir, args.overwrite)
        groups.to_csv(destination / "topology_groups.csv", index=False)
        all_summaries = {}
        for task in selected_tasks(args.task):
            print(f"Stage 1: {task}, {len(frame)} samples", flush=True)
            predictor = load_predictor(task, args.models_dir, args.device)

            def predict(sequences):
                # Masked sequences are intentionally not complete structures.
                return predictor.predict(sequences, batch_size=args.batch_size, validate=False)

            local = pd.DataFrame(exact_topology_attributions(
                frame["MOFseq"].tolist(), predict, frame["NAME"].tolist()
            ))
            # Aggregate by the actual normalized <T> genome, never a display
            # label. Map names only after counts and SHAP statistics are fixed.
            if local["topology_genome"].tolist() != genomes.tolist():
                raise ValueError("Attribution identities differ from the validated input genomes.")
            local["feature_key"] = local["topology_genome"]
            local["feature_group"] = "topology"
            local["shap_value"] = local["topology_shap"]
            values, summary = paper_tables(local, minimum=15, top_n=10)
            all_summaries[task] = paper_tables(local, minimum=1, top_n=len(frame))[1]
            values["feature_key"] = values["feature_key"].map(display_names)
            summary["feature_key"] = summary["feature_key"].map(display_names)

            target_dir = destination / task
            target_dir.mkdir(exist_ok=True)
            values.to_csv(target_dir / "shap_values.csv", index=False)
            summary.to_csv(target_dir / "summary.csv", index=False)
            print(f"Saved top 10 common topologies (n >= 15): {target_dir}", flush=True)
        if set(all_summaries) == {"selectivity", "ch4_uptake"}:
            # Table S9 compares both targets for the selectivity-ranked top ten.
            ranked = all_summaries["selectivity"]
            ranked = ranked.loc[ranked["n"].ge(15)].head(10).copy()
            uptake = all_summaries["ch4_uptake"].set_index("feature_key")
            table = pd.DataFrame({
                "Topology": ranked["feature_key"].map(display_names),
                "n": ranked["n"],
                "Mean_SHAP_selectivity": ranked["mean_shap"],
                "Positive_selectivity_percent": ranked["positive_fraction"] * 100,
                "Mean_SHAP_CH4_uptake": ranked["feature_key"].map(uptake["mean_shap"]),
                "Positive_CH4_uptake_percent": ranked["feature_key"].map(uptake["positive_fraction"]) * 100,
            })
            table.to_csv(destination / "table_S9.csv", index=False)
            print(f"Saved both-target comparison for Table S9: {destination / 'table_S9.csv'}", flush=True)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
