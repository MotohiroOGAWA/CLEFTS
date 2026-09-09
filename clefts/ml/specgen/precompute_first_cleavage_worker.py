from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from clefts.libs.msentity.msentity import MSDataset
from clefts.ml.input.fragment_tree_preprocessing_context import (
    FragmentTreePreprocessingContext,
)
from clefts.ml.input.single_fragment_tree_structure_builder import (
    SingleFragmentTreeStructureBuilder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build precursor/first-cleavage caches for one prediction batch."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--task-json", required=True)
    parser.add_argument("--context-json", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--smiles-column", required=True)
    parser.add_argument("--precursor-mz-column", required=True)
    parser.add_argument("--adduct-type-column", required=True)
    parser.add_argument("--collision-energy-column", required=True)
    parser.add_argument("--instrument-column")
    parser.add_argument("--spec-id-column", required=True)
    return parser.parse_args()


def _metadata(record: object) -> dict[str, Any]:
    return {column: record[column] for column in record.columns}


def main() -> None:
    args = parse_args()
    dataset = MSDataset.load(args.input)
    context_data = json.loads(Path(args.context_json).read_text(encoding="utf-8"))
    context = FragmentTreePreprocessingContext(**context_data)
    tasks = json.loads(Path(args.task_json).read_text(encoding="utf-8"))
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for task in tasks:
        index, smiles = int(task["index"]), str(task["smiles"])
        sub_dataset = dataset[dataset[args.smiles_column].astype(str) == smiles]
        source_ids = sub_dataset[args.spec_id_column].astype(str).tolist()
        compound_failures: list[dict[str, str]] = []
        try:
            builder = SingleFragmentTreeStructureBuilder(context)
            sample_indexes = builder.add_same_smiles_dataset_first_cleavage(
                sub_dataset,
                precursor_mz_column=args.precursor_mz_column,
                adduct_type_column=args.adduct_type_column,
                collision_energy_column=args.collision_energy_column,
                smiles_column=args.smiles_column,
                instrument_column=args.instrument_column,
            )
            metadata_by_sample: dict[int, dict[str, Any]] = {}
            for record_index, sample_index in enumerate(sample_indexes.tolist()):
                if int(sample_index) < 0:
                    failure = {
                        "SourceSpecID": source_ids[record_index],
                        "FailureType": "InputRejected",
                        "FailureMessage": "No valid precursor pathway was generated.",
                    }
                    failures.append(failure)
                    compound_failures.append(failure)
                    continue
                metadata_by_sample[int(sample_index)] = _metadata(
                    sub_dataset[record_index]
                )
            if not metadata_by_sample or not builder.node_graph:
                raise ValueError("No valid first-cleavage samples were generated.")
            metadata = pd.DataFrame([
                metadata_by_sample[sample_index]
                for sample_index in sorted(metadata_by_sample)
            ])
            target = cache_dir / f"compound-{index:09d}.pt"
            torch.save({
                "schema": "clefts.first-cleavage-cache",
                "schema_version": 1,
                "smiles": smiles,
                "builder_state": builder.export_state(),
                "metadata": metadata,
            }, target)
            completed.append({
                "index": index,
                "path": str(target),
                "sample_count": len(sub_dataset),
            })
        except Exception as exc:
            rejected_ids = {row["SourceSpecID"] for row in compound_failures}
            for source_id in source_ids:
                if source_id not in rejected_ids:
                    failures.append({
                        "SourceSpecID": source_id,
                        "FailureType": type(exc).__name__,
                        "FailureMessage": str(exc),
                    })

    Path(args.result_json).write_text(json.dumps({
        "completed": completed,
        "failures": failures,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
