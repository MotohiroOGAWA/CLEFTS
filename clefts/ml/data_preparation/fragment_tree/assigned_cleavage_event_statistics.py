from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from clefts.domain.fragment.cleavage import CleavagePatternSet
from clefts.libs.mmkit.mmkit import Compound

_BOND_SYMBOL = {
    Chem.BondType.SINGLE: "-",
    Chem.BondType.DOUBLE: "=",
    Chem.BondType.TRIPLE: "#",
    Chem.BondType.AROMATIC: ":",
}


def _matched_substructure_label(compound: Compound, atom_indexes: tuple[int, ...]) -> str:
    mol = compound.mol
    if len(atom_indexes) == 2:
        first, second = atom_indexes
        first_symbol = mol.GetAtomWithIdx(first).GetSymbol()
        second_symbol = mol.GetAtomWithIdx(second).GetSymbol()
        bond = mol.GetBondBetweenAtoms(first, second)
        bond_symbol = _BOND_SYMBOL.get(bond.GetBondType(), "~") if bond is not None else "."
        forward = f"{first_symbol}{bond_symbol}{second_symbol}"
        reverse = f"{second_symbol}{bond_symbol}{first_symbol}"
        return min(forward, reverse)
    return Chem.MolFragmentToSmiles(
        mol, atomsToUse=list(atom_indexes), canonical=True, isomericSmiles=False
    )


def _collect_structure_statistics(
    structure_file_str: str,
) -> dict[str, Counter]:
    """Collect assigned-event counts for one structure; safe for subprocess use."""
    from clefts.ml.input.fragment_tree_training_data import (
        load_fragment_tree_structure_file,
    )

    structure_file = Path(structure_file_str)
    item = load_fragment_tree_structure_file(structure_file, map_location="cpu")
    structure = item.structure
    event_counts: Counter[tuple[int, str]] = Counter()
    pathway_counts: Counter[tuple[int, str]] = Counter()
    sample_counts: Counter[tuple[int, int, str, int, str]] = Counter()
    pattern_event_counts: Counter[int] = Counter()
    pattern_pathway_counts: Counter[int] = Counter()
    pattern_sample_presence: Counter[tuple[int, int]] = Counter()
    reaction_event_counts: Counter[tuple[int, int]] = Counter()
    reaction_pathway_counts: Counter[tuple[int, int]] = Counter()
    reaction_sample_presence: Counter[tuple[int, int, int]] = Counter()
    product_event_counts: Counter[tuple[int, int, int]] = Counter()
    product_pathway_counts: Counter[tuple[int, int, int]] = Counter()
    product_sample_presence: Counter[tuple[int, int, int, int]] = Counter()
    empty_result = {
        "events": event_counts, "pathways": pathway_counts, "samples": sample_counts,
        "pattern_events": pattern_event_counts,
        "pattern_pathways": pattern_pathway_counts,
        "pattern_samples": pattern_sample_presence,
        "reaction_events": reaction_event_counts,
        "reaction_pathways": reaction_pathway_counts,
        "reaction_samples": reaction_sample_presence,
        "product_events": product_event_counts,
        "product_pathways": product_pathway_counts,
        "product_samples": product_sample_presence,
    }
    if not hasattr(structure, "terminal_expand_ptr"):
        return empty_result

    edge_by_nodes = {
        (int(source), int(target)): edge_index
        for edge_index, (source, target) in enumerate(structure.edge_index.t().tolist())
    }
    event_indexes_by_edge: dict[int, list[int]] = {}
    for event_index, edge_index in enumerate(structure.cleavage_event_edge_index.tolist()):
        event_indexes_by_edge.setdefault(int(edge_index), []).append(event_index)
    tuple_length_by_pattern_reaction = {
        (int(pattern_id), int(reaction_id)): int(tuple_length)
        for pattern_id, reaction_id, tuple_length
        in structure.reactant_tuple_length_table.tolist()
    }
    saved_sample_indexes = item.metadata.get("sample_indexes", [])
    source_record_indexes = item.metadata.get(
        "source_record_indexes", item.metadata.get("record_indexes", [])
    )
    record_index_by_sample = {
        int(sample_index): int(record_index)
        for record_index, sample_index in zip(source_record_indexes, saved_sample_indexes)
    }
    spec_id_by_sample = {
        int(sample_index): str(spec_id)
        for spec_id, sample_index in zip(
            item.metadata.get("spec_ids", []), saved_sample_indexes
        )
        if str(spec_id)
    }
    compound_by_node: dict[int, Compound] = {}

    for assignment_index, sample_index_value in enumerate(
        structure.target_sample_index.tolist()
    ):
        sample_index = int(sample_index_value)
        start = int(structure.terminal_expand_ptr[assignment_index])
        stop = int(structure.terminal_expand_ptr[assignment_index + 1])
        nodes = [
            int(value)
            for value in structure.target_expand_node_index[start:stop].tolist()
        ]
        nodes.append(int(structure.target_terminal_node_index[assignment_index]))
        pathway_types: set[tuple[int, str]] = set()
        pathway_patterns: set[int] = set()
        pathway_reactions: set[tuple[int, int]] = set()
        pathway_products: set[tuple[int, int, int]] = set()

        for source_node, target_node in zip(nodes, nodes[1:]):
            edge_index = edge_by_nodes.get((source_node, target_node))
            if edge_index is None:
                continue
            if source_node not in compound_by_node:
                compound_by_node[source_node] = Compound.from_smiles(
                    str(structure.node_smiles[source_node])
                )
            compound = compound_by_node[source_node]
            for event_index in event_indexes_by_edge.get(edge_index, []):
                event = structure.cleavage_event[event_index]
                pattern_id = int(event[0])
                reaction_id = int(event[1])
                product_molecule_id = int(event[2])
                tuple_length = tuple_length_by_pattern_reaction[(pattern_id, reaction_id)]
                atom_row_index = int(event[3])
                atom_indexes = tuple(
                    int(value)
                    for value in structure.cleavage_atom_idxs[tuple_length][atom_row_index].tolist()
                )
                label = _matched_substructure_label(compound, atom_indexes)
                key = (pattern_id, label)
                event_counts[key] += 1
                sample_counts[
                    sample_index,
                    record_index_by_sample.get(sample_index, -1),
                    spec_id_by_sample.get(sample_index, ""),
                    pattern_id,
                    label,
                ] += 1
                reaction_key = (pattern_id, reaction_id)
                product_key = (pattern_id, reaction_id, product_molecule_id)
                pattern_event_counts[pattern_id] += 1
                reaction_event_counts[reaction_key] += 1
                product_event_counts[product_key] += 1
                pattern_sample_presence[(sample_index, pattern_id)] = 1
                reaction_sample_presence[(sample_index, *reaction_key)] = 1
                product_sample_presence[(sample_index, *product_key)] = 1
                pathway_types.add(key)
                pathway_patterns.add(pattern_id)
                pathway_reactions.add(reaction_key)
                pathway_products.add(product_key)
        for key in pathway_types:
            pathway_counts[key] += 1
        for pattern_id in pathway_patterns:
            pattern_pathway_counts[pattern_id] += 1
        for key in pathway_reactions:
            reaction_pathway_counts[key] += 1
        for key in pathway_products:
            product_pathway_counts[key] += 1

    return empty_result


def _collect_all(
    structure_files: list[Path], num_workers: int
) -> Iterable[tuple[Path, dict[str, Counter]]]:
    if num_workers <= 1:
        for path in structure_files:
            yield path, _collect_structure_statistics(str(path))
        return
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=get_context("spawn"),
    ) as executor:
        results = executor.map(
            _collect_structure_statistics,
            (str(path) for path in structure_files),
            chunksize=8,
        )
        for path, result in zip(structure_files, results):
            yield path, result


def write_assigned_cleavage_event_statistics(
    *,
    structure_files: list[Path],
    pattern_set: CleavagePatternSet,
    output_file: str | Path,
    num_workers: int = 1,
) -> None:
    """Count events in peak-assigned FragmentPathways separately per sample."""
    patterns = {int(pattern.pattern_id): pattern for pattern in pattern_set.patterns}
    event_counts: Counter[tuple[int, str]] = Counter()
    pathway_counts: Counter[tuple[int, str]] = Counter()
    samples_by_type: dict[tuple[int, str], set[tuple[str, int]]] = {}
    pattern_event_counts: Counter[int] = Counter()
    pattern_pathway_counts: Counter[int] = Counter()
    pattern_sample_counts: Counter[int] = Counter()
    reaction_event_counts: Counter[tuple[int, int]] = Counter()
    reaction_pathway_counts: Counter[tuple[int, int]] = Counter()
    reaction_sample_counts: Counter[tuple[int, int]] = Counter()
    product_event_counts: Counter[tuple[int, int, int]] = Counter()
    product_pathway_counts: Counter[tuple[int, int, int]] = Counter()
    product_sample_counts: Counter[tuple[int, int, int]] = Counter()
    sample_rows = []

    results = _collect_all(structure_files, num_workers)
    for structure_file, result in tqdm(
        results,
        total=len(structure_files),
        desc="Counting assigned cleavage events",
        mininterval=1.0,
    ):
        event_counts.update(result["events"])
        pathway_counts.update(result["pathways"])
        pattern_event_counts.update(result["pattern_events"])
        pattern_pathway_counts.update(result["pattern_pathways"])
        reaction_event_counts.update(result["reaction_events"])
        reaction_pathway_counts.update(result["reaction_pathways"])
        product_event_counts.update(result["product_events"])
        product_pathway_counts.update(result["product_pathways"])
        for _, pattern_id in result["pattern_samples"]:
            pattern_sample_counts[pattern_id] += 1
        for _, pattern_id, reaction_id in result["reaction_samples"]:
            reaction_sample_counts[pattern_id, reaction_id] += 1
        for _, pattern_id, reaction_id, product_molecule_id in result["product_samples"]:
            product_sample_counts[pattern_id, reaction_id, product_molecule_id] += 1
        for (sample_index, record_index, spec_id, pattern_id, label), count in result["samples"].items():
            key = (pattern_id, label)
            samples_by_type.setdefault(key, set()).add((structure_file.name, sample_index))
            pattern = patterns[pattern_id]
            sample_rows.append({
                "record_index": record_index,
                "SpecID": spec_id,
                "structure_file": structure_file.name,
                "structure_sample_index": sample_index,
                "reactant_smarts": pattern.reactant_smarts,
                "matched_substructure": label,
                "assigned_cleavage_event_count": count,
                "pattern_id": pattern_id,
            })

    rows = []
    for (pattern_id, label), event_count in event_counts.items():
        pattern = patterns[pattern_id]
        rows.append({
            "reactant_smarts": pattern.reactant_smarts,
            "matched_substructure": label,
            "assigned_cleavage_event_count": event_count,
            "assigned_pathway_count": pathway_counts[pattern_id, label],
            "sample_count": len(samples_by_type.get((pattern_id, label), set())),
            "pattern_id": pattern_id,
        })
    rows.sort(key=lambda row: (
        -int(row["assigned_cleavage_event_count"]),
        str(row["reactant_smarts"]),
        str(row["matched_substructure"]),
    ))
    columns = [
        "reactant_smarts", "matched_substructure", "assigned_cleavage_event_count",
        "assigned_pathway_count", "sample_count", "pattern_id",
    ]
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(output_path, sep="\t", index=False)

    sample_rows.sort(key=lambda row: (
        int(row["record_index"]), str(row["structure_file"]),
        int(row["structure_sample_index"]),
        -int(row["assigned_cleavage_event_count"]), str(row["matched_substructure"]),
    ))
    sample_columns = [
        "record_index", "SpecID", "structure_file", "structure_sample_index", "reactant_smarts",
        "matched_substructure", "assigned_cleavage_event_count", "pattern_id",
    ]
    sample_output_path = output_path.with_name(
        f"{output_path.stem}_by_sample{output_path.suffix}"
    )
    pd.DataFrame(sample_rows, columns=sample_columns).to_csv(
        sample_output_path, sep="\t", index=False
    )

    pattern_rows = [{
        "pattern_id": pattern_id,
        "reactant_smarts": patterns[pattern_id].reactant_smarts,
        "assigned_cleavage_event_count": count,
        "assigned_pathway_count": pattern_pathway_counts[pattern_id],
        "sample_count": pattern_sample_counts[pattern_id],
    } for pattern_id, count in pattern_event_counts.items()]
    pattern_rows.sort(key=lambda row: (
        -int(row["assigned_cleavage_event_count"]), int(row["pattern_id"])
    ))
    pattern_output_path = output_path.with_name(
        f"{output_path.stem}_by_pattern{output_path.suffix}"
    )
    pattern_columns = [
        "pattern_id", "reactant_smarts", "assigned_cleavage_event_count",
        "assigned_pathway_count", "sample_count",
    ]
    pd.DataFrame(pattern_rows, columns=pattern_columns).to_csv(
        pattern_output_path, sep="\t", index=False
    )

    reaction_rows = [{
        "pattern_id": pattern_id,
        "reaction_id": reaction_id,
        "reactant_smarts": patterns[pattern_id].reactant_smarts,
        "assigned_cleavage_event_count": count,
        "assigned_pathway_count": reaction_pathway_counts[pattern_id, reaction_id],
        "sample_count": reaction_sample_counts[pattern_id, reaction_id],
    } for (pattern_id, reaction_id), count in reaction_event_counts.items()]
    reaction_rows.sort(key=lambda row: (
        -int(row["assigned_cleavage_event_count"]),
        int(row["pattern_id"]), int(row["reaction_id"]),
    ))
    reaction_output_path = output_path.with_name(
        f"{output_path.stem}_by_pattern_reaction{output_path.suffix}"
    )
    reaction_columns = [
        "pattern_id", "reaction_id", "reactant_smarts",
        "assigned_cleavage_event_count", "assigned_pathway_count", "sample_count",
    ]
    pd.DataFrame(reaction_rows, columns=reaction_columns).to_csv(
        reaction_output_path, sep="\t", index=False
    )

    product_rows = [{
        "pattern_id": pattern_id,
        "reaction_id": reaction_id,
        "product_molecule_id": product_molecule_id,
        "reactant_smarts": patterns[pattern_id].reactant_smarts,
        "assigned_cleavage_event_count": count,
        "assigned_pathway_count": product_pathway_counts[
            pattern_id, reaction_id, product_molecule_id
        ],
        "sample_count": product_sample_counts[
            pattern_id, reaction_id, product_molecule_id
        ],
    } for (pattern_id, reaction_id, product_molecule_id), count in product_event_counts.items()]
    product_rows.sort(key=lambda row: (
        -int(row["assigned_cleavage_event_count"]),
        int(row["pattern_id"]), int(row["reaction_id"]),
        int(row["product_molecule_id"]),
    ))
    product_output_path = output_path.with_name(
        f"{output_path.stem}_by_pattern_reaction_product{output_path.suffix}"
    )
    product_columns = [
        "pattern_id", "reaction_id", "product_molecule_id", "reactant_smarts",
        "assigned_cleavage_event_count", "assigned_pathway_count",
        "sample_count",
    ]
    pd.DataFrame(product_rows, columns=product_columns).to_csv(
        product_output_path, sep="\t", index=False
    )
    print(f"saved assigned cleavage event statistics: {output_path}")
    print(f"saved per-sample cleavage event statistics: {sample_output_path}")


def _load_pattern_set(params_file: str | Path) -> CleavagePatternSet:
    with Path(params_file).open("r", encoding="utf-8") as file:
        data = json.load(file)
    params = data.get("probability_model_params", data)
    fragmenter = params.get("fragmenter_params", params)
    builder = fragmenter.get("fragment_ion_tree_builder", fragmenter)
    return CleavagePatternSet.from_dict(builder["cleavage_pattern_set"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count cleavage events in peak-assigned FragmentPathways per sample."
    )
    parser.add_argument("--structures-dir", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    structures_dir = Path(args.structures_dir)
    if not any(structures_dir.glob("*.preft")) and not any(structures_dir.glob("*.pt")):
        structures_dir = structures_dir / "data"
    structure_files = sorted(set(structures_dir.glob("*.preft")) | set(structures_dir.glob("*.pt")))
    write_assigned_cleavage_event_statistics(
        structure_files=structure_files,
        pattern_set=_load_pattern_set(args.params),
        output_file=args.output,
        num_workers=max(1, args.num_workers),
    )


if __name__ == "__main__":
    main()
