"""Encode linker SMILES and assemble already identified fields into MOFseq.

Examples:
    python MOFseq_encoding.py encode --input data/linkers.csv --output results/linkers_encoded.csv
    python MOFseq_encoding.py assemble --input fields.csv --output results/candidates.csv

Encoding dependencies are listed in requirements.txt. Assembly is not a CIF parser:
the topology, interpenetration and node environment must already be identified.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import re
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd
from model import ModelConfig
from utils import parse_mofseq, validate_parsed_mofseq

ROOT = Path(__file__).resolve().parent


def load_chemistry() -> None:
    """Load linker-encoding packages only when the encode command is used."""
    global nx, sf, Group, GroupGrammar, GraphMatcher, categorical_node_match, Chem, RDLogger
    try:
        import networkx as nx
        import selfies as sf
        from group_selfies import Group, GroupGrammar
        from networkx.algorithms.isomorphism import GraphMatcher, categorical_node_match
        from rdkit import Chem, RDLogger
    except ImportError as exc:
        raise ImportError(
            "Linker-encoding packages are missing. "
            "Run: python -m pip install -r requirements.txt"
        ) from exc


DEFAULT_SEED = 20260804
EXTRA_MATCH_VARIANTS = {
    "benzimidazole": ["C1(*1)=C(*1)C2=C(N=C(*1)N2*1)C(*1)=C1*1"],
}
EMPTY_VALUES = {"", "0", "0.0", "nan", "none", "null", "na", "n/a", "<na>"}


@dataclass(frozen=True)
class AliasSpec:
    canonical_name: str
    mappings: tuple[Mapping[int, int], ...]


@dataclass
class GrammarBundle:
    output_grammar: GroupGrammar
    matching_grammar: GroupGrammar
    aliases: dict[str, AliasSpec]
    primary_count: int
    variant_count: int


@dataclass
class GroupCandidate:
    encoded: str
    roundtrip: str
    covered_atoms: int
    group_count: int
    largest_group: int
    priority_score: int
    group_names: str

    @property
    def score(self) -> tuple[int, int, int, int, int]:
        return (self.covered_atoms, self.priority_score, self.largest_group,
                self.group_count, 2 if self.roundtrip == "EXACT" else 1)


def clean_smiles(value: str | None) -> str | None:
    if value is None:
        return None
    value = re.sub(r"\s+", "", value.strip())
    return None if value.lower() in EMPTY_VALUES else value


def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def chemical_identity(mol: Chem.Mol) -> tuple[str, str, int]:
    try:
        key = Chem.MolToInchiKey(mol) or ""
    except Exception:
        key = ""
    return canonical_smiles(mol), key, Chem.GetFormalCharge(mol)


def compare_identity(reference: tuple[str, str, int], decoded: Chem.Mol | None) -> str:
    if decoded is None:
        return "MISMATCH"
    actual = chemical_identity(decoded)
    if actual[2] != reference[2]:
        return "MISMATCH"
    if actual[0] == reference[0]:
        return "EXACT"
    if reference[1] and actual[1] == reference[1]:
        return "INCHI_EQUIVALENT"
    return "MISMATCH"


def parse_fragment_line(line: str) -> tuple[str, str] | None:
    line = line.strip().replace("\u3000", " ")
    if not line or line.startswith("#"):
        return None
    fields = line.split("\t", 1) if "\t" in line else re.split(r"\s+", line, maxsplit=1)
    if len(fields) != 2:
        raise ValueError("Expected a fragment name and structure separated by a tab.")
    name, structure = fields[0].strip(), re.sub(r"\s+", "", fields[1])
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or not structure:
        raise ValueError("Invalid fragment name or empty structure.")
    return name, structure


def group_topology_graph(group: Group) -> nx.Graph:
    """Map variants without changing elements, charge, or attachment identities."""
    graph = nx.Graph()
    for atom in group.pattern.GetAtoms():
        attachment = bool(atom.HasProp("valAvailable"))
        graph.add_node(atom.GetIdx(), atomic_number=int(atom.GetAtomicNum()),
                       formal_charge=int(atom.GetFormalCharge()), isotope=int(atom.GetIsotope()),
                       attachment=attachment,
                       attachment_valence=int(atom.GetIntProp("valAvailable")) if attachment else 0)
    for bond in group.pattern.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return graph


def find_variant_mappings(variant: Group, canonical: Group, maximum: int = 64) -> tuple[Mapping[int, int], ...]:
    match = categorical_node_match(
        ["atomic_number", "formal_charge", "isotope", "attachment", "attachment_valence"],
        [0, 0, 0, False, 0],
    )
    matcher = GraphMatcher(group_topology_graph(variant), group_topology_graph(canonical), node_match=match)
    mappings, seen = [], set()
    for mapping in matcher.isomorphisms_iter():
        signature = tuple(sorted((int(k), int(v)) for k, v in mapping.items()))
        if signature not in seen:
            seen.add(signature)
            mappings.append(dict(signature))
        if len(mappings) >= maximum:
            break
    return tuple(mappings)


def load_grammar_bundle(fragment_file: Path) -> GrammarBundle:
    """First definition per name is canonical; repeated names are match variants."""
    definitions = []
    with Path(fragment_file).open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                parsed = parse_fragment_line(line)
                if parsed is not None:
                    definitions.append(parsed)
            except ValueError as exc:
                raise ValueError(f"Fragment dictionary line {line_number}: {exc}") from exc
    primary, primary_smiles, pending = {}, {}, []
    for name, fragment in definitions:
        if name in primary:
            pending.append((name, fragment))
        else:
            try:
                primary[name] = Group(name, fragment)
                primary_smiles[name] = fragment
            except Exception as exc:
                raise ValueError(f"Could not parse canonical fragment {name!r}: {exc}") from exc
    if not primary:
        raise ValueError("Fragment dictionary contains no groups.")
    for name, variants in EXTRA_MATCH_VARIANTS.items():
        if name in primary:
            pending.extend((name, fragment) for fragment in variants if fragment != primary_smiles[name])
    matching, aliases = dict(primary), {}
    seen = set(primary_smiles.items())
    counter: dict[str, int] = {}
    for name, fragment in pending:
        if (name, fragment) in seen:
            continue
        seen.add((name, fragment))
        counter[name] = counter.get(name, 0) + 1
        internal_name = f"matchvariant_{name}_{counter[name]}"
        try:
            group = Group(internal_name, fragment)
            mappings = find_variant_mappings(group, primary[name])
            if not mappings:
                raise ValueError("Variant atoms/attachment sites do not match the canonical group.")
            matching[internal_name] = group
            aliases[internal_name] = AliasSpec(name, mappings)
        except Exception as exc:
            raise ValueError(f"Could not load matching variant for {name!r}: {exc}") from exc
    return GrammarBundle(GroupGrammar(vocab=primary), GroupGrammar(vocab=matching),
                         aliases, len(primary), len(aliases))


def translate_group_options(extracted: Sequence, bundle: GrammarBundle, maximum: int = 64) -> list[list]:
    choices = []
    for group, atoms, bonds in extracted:
        alias = bundle.aliases.get(group.name)
        if alias is None:
            choices.append([(group, atoms, bonds)])
            continue
        variants = []
        for mapping in alias.mappings:
            try:
                canonical = bundle.output_grammar.get_group(alias.canonical_name)
                mapped_atoms = [(outer, mapping[inner]) for outer, inner in atoms]
                mapped_bonds = [(outer, mapping[inner]) for outer, inner in bonds]
            except KeyError:
                continue
            variants.append((canonical, mapped_atoms, mapped_bonds))
        if not variants:
            return []
        choices.append(variants)
    return [list(parts) for parts in itertools.islice(itertools.product(*choices), maximum)] or [[]]


def candidate_smiles_strings(mol: Chem.Mol, original: str, random_trials: int, seed: int) -> tuple[list[str], list[str]]:
    base = [canonical_smiles(mol), original]
    try:
        base.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True, kekuleSmiles=True))
    except Exception:
        pass
    random = []
    if random_trials:
        molecule_seed = (seed + zlib.crc32(canonical_smiles(mol).encode("utf-8"))) % (2**31 - 1)
        for trials, offset, kekule in (((random_trials+1)//2, 0, False), (random_trials//2, 104729, True)):
            if trials:
                try:
                    random.extend(Chem.MolToRandomSmilesVect(mol, trials, int((molecule_seed+offset) % (2**31-1)), True, kekule))
                except Exception:
                    pass
    def unique(values: Iterable[str]) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))
    base = unique(base)
    return base, unique(value for value in random if value not in set(base))


def evaluate_group_candidate(smiles: str, reference: tuple[str, str, int], bundle: GrammarBundle) -> list[GroupCandidate]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    try:
        extracted = bundle.matching_grammar.extract_groups(mol)
    except Exception:
        return []
    results = []
    for translated in translate_group_options(extracted, bundle):
        try:
            encoded = bundle.output_grammar.encoder(Chem.MolFromSmiles(smiles), translated)
            roundtrip = compare_identity(reference, bundle.output_grammar.decoder(encoded))
            if roundtrip == "MISMATCH":
                continue
            covered = {int(outer) for _, atoms, _ in translated for outer, _ in atoms}
            sizes = [len(atoms) for _, atoms, _ in translated]
            priority = sum(int(getattr(group, "priority", 0))*len(atoms) for group, atoms, _ in translated)
            results.append(GroupCandidate(encoded, roundtrip, len(covered), len(translated),
                                          max(sizes, default=0), priority,
                                          ";".join(sorted(group.name for group, _, _ in translated))))
        except Exception:
            continue
    return results


def robust_group_selfies(mol: Chem.Mol, original: str, bundle: GrammarBundle, random_trials: int = 16, seed: int = DEFAULT_SEED) -> tuple[GroupCandidate | None, int]:
    reference = chemical_identity(mol)
    base, random = candidate_smiles_strings(mol, original, random_trials, seed)
    candidates, tested = [], 0
    for smiles in base:
        tested += 1
        candidates.extend(evaluate_group_candidate(smiles, reference, bundle))
    heavy_atoms = sum(atom.GetAtomicNum() > 1 for atom in mol.GetAtoms())
    if max((candidate.covered_atoms for candidate in candidates), default=-1) < heavy_atoms:
        for smiles in random:
            tested += 1
            candidates.extend(evaluate_group_candidate(smiles, reference, bundle))
    if not candidates:
        return None, tested
    best = max(candidate.score for candidate in candidates)
    return next(candidate for candidate in candidates if candidate.score == best), tested


RESULT_COLUMNS = ["Canonical_SMILES", "GroupSELFIES", "GroupSELFIES_tokens", "GroupSELFIES_status",
                  "GroupSELFIES_roundtrip", "GroupSELFIES_decoded_SMILES", "Formal_charge",
                  "Decoded_formal_charge", "GroupSELFIES_group_names", "GroupSELFIES_covered_atoms",
                  "GroupSELFIES_group_count", "GroupSELFIES_candidates_tested", "GroupSELFIES_error"]


def encode_one(smiles_value: str | None, bundle: GrammarBundle, random_trials: int = 16, seed: int = DEFAULT_SEED) -> dict:
    result = dict.fromkeys(RESULT_COLUMNS, "")
    smiles = clean_smiles(smiles_value)
    if smiles is None:
        result["GroupSELFIES_status"] = "EMPTY"
        return result
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        result.update(GroupSELFIES_status="INVALID_SMILES", GroupSELFIES_error="RDKit could not parse this SMILES.")
        return result
    result.update(Canonical_SMILES=canonical_smiles(mol), Formal_charge=Chem.GetFormalCharge(mol))
    candidate, tested = robust_group_selfies(mol, smiles, bundle, random_trials, seed)
    result["GroupSELFIES_candidates_tested"] = tested
    if candidate is None:
        result.update(GroupSELFIES_status="ENCODING_OR_ROUNDTRIP_FAILED", GroupSELFIES_roundtrip="MISMATCH",
                      GroupSELFIES_error="No reversible Group SELFIES encoding was found.")
        return result
    decoded = bundle.output_grammar.decoder(candidate.encoded)
    result.update(GroupSELFIES=candidate.encoded, GroupSELFIES_tokens=" ".join(sf.split_selfies(candidate.encoded)),
                  GroupSELFIES_status="SUCCESS", GroupSELFIES_roundtrip=candidate.roundtrip,
                  GroupSELFIES_decoded_SMILES=canonical_smiles(decoded), Decoded_formal_charge=Chem.GetFormalCharge(decoded),
                  GroupSELFIES_group_names=candidate.group_names, GroupSELFIES_covered_atoms=candidate.covered_atoms,
                  GroupSELFIES_group_count=candidate.group_count)
    return result



def bracket_tokens(value: str) -> str:
    compact = re.sub(r"\s+", "", value)
    tokens = re.findall(r"\[[^\[\]\s]+\]", compact)
    if not tokens or "".join(tokens) != compact:
        raise ValueError("Expected a complete bracket-token string")
    return " ".join(tokens)


def assemble_row(row: dict) -> str:
    genome = " ".join(str(row["topology_genome"]).split())
    cat = str(row["interpenetration"]).strip()
    if not re.fullmatch(r"\d+", cat) or int(cat) < 1:
        raise ValueError("interpenetration must be a positive integer")
    if any(not re.fullmatch(r"-?\d+", value) for value in genome.split()):
        raise ValueError("topology_genome must contain space-separated integers")
    node = " ".join(str(row["nodeid"]).split())
    if not node or "<" in node or ">" in node:
        raise ValueError("nodeid must contain node content, not MOFseq field markers")
    linker = str(row["GroupSELFIES_tokens"]).strip()
    if not linker:
        raise ValueError("GroupSELFIES_tokens must not be empty")
    linker = bracket_tokens(linker)
    sequence = f"<T> {genome} <cat> {cat} <N1> {node} <L1> {linker}"
    ion = str(row.get("ion", "")).strip()
    if ion and ion.lower() != "nan":
        sequence += f" <I1> {bracket_tokens(ion)}"
    sequence = " ".join(sequence.split())
    validate_parsed_mofseq(parse_mofseq(sequence), ModelConfig())
    return sequence



def main_encode(args: argparse.Namespace) -> int:
    load_chemistry()
    if args.random_trials < 0:
        raise ValueError("--random-trials cannot be negative.")
    if not args.show_rdkit_messages:
        RDLogger.DisableLog("rdApp.*")
    bundle = load_grammar_bundle(args.fragments)
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or args.smiles_column not in reader.fieldnames:
            raise ValueError(f"Input CSV must contain a {args.smiles_column!r} column.")
        columns = list(reader.fieldnames)
        rows = list(reader)
    if not rows:
        raise ValueError("Input CSV has no records.")
    collision = sorted(set(columns).intersection(RESULT_COLUMNS))
    if collision:
        raise ValueError(f"Input contains reserved output columns: {collision}")
    cache, output, failures = {}, [], 0
    for row in rows:
        cleaned = clean_smiles(row[args.smiles_column])
        probe = Chem.MolFromSmiles(cleaned) if cleaned else None
        key = canonical_smiles(probe) if probe is not None else "<EMPTY>" if cleaned is None else "<BAD>"+cleaned
        if key not in cache:
            cache[key] = encode_one(row[args.smiles_column], bundle, args.random_trials, args.seed)
        result = cache[key]
        failures += result["GroupSELFIES_status"] not in {"SUCCESS", "EMPTY"}
        output.append({**row, **result})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns+RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(output)
    print(f"Groups: {bundle.primary_count}; matching variants: {bundle.variant_count}; "
          f"rows: {len(rows)}; failed rows: {failures}; output: {args.output}")
    return 1 if failures else 0


def main_assemble(args: argparse.Namespace) -> int:
    frame = pd.read_csv(args.input, keep_default_na=False, dtype=str)
    required = {"topology_genome", "interpenetration", "nodeid", "GroupSELFIES_tokens"}
    if not required.issubset(frame):
        raise ValueError(f"Required input columns: {sorted(required)}")
    if frame.empty:
        raise ValueError("Input CSV has no records.")
    frame["MOFseq"] = [assemble_row(row) for row in frame.to_dict("records")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, mode="x")
    print(f"Wrote {len(frame)} assembled sequences to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    encode = sub.add_parser("encode", help="Convert linker SMILES to Group SELFIES.")
    encode.add_argument("--input", type=Path, default=ROOT / "data/linkers.csv")
    encode.add_argument("--output", type=Path, default=ROOT / "results/linkers_encoded.csv")
    encode.add_argument("--fragments", type=Path, default=ROOT / "data/fragments.tsv")
    encode.add_argument("--smiles-column", default="SMILES")
    encode.add_argument("--random-trials", type=int, default=16)
    encode.add_argument("--seed", type=int, default=DEFAULT_SEED)
    encode.add_argument("--show-rdkit-messages", action="store_true")
    encode.set_defaults(handler=main_encode)
    assemble = sub.add_parser("assemble", help="Combine topology, node and linker fields.")
    assemble.add_argument("--input", type=Path, required=True)
    assemble.add_argument("--output", type=Path, default=ROOT / "results/candidates.csv")
    assemble.set_defaults(handler=main_assemble)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.input.resolve() == args.output.resolve() or args.output.exists():
            raise ValueError("Use a new output file; existing files are never overwritten.")
        return args.handler(args)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())

