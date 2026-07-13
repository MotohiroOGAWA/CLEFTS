from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from clefts.domain.fragment.cleavage import CleavagePatternSet
from clefts.libs.mmkit.mmkit import Compound
from clefts.libs.msentity.msentity import MSDataset

CLASSIFICATION_COLUMNS = ("Kingdom", "Superclass", "Class", "Subclass", "DirectParent")


def write_cleavage_pattern_statistics(*, dataset: MSDataset, pattern_set: CleavagePatternSet,
                                      output_dir: str | Path, split_name: str,
                                      smiles_column: str) -> None:
    """Write reactant-SMARTS coverage statistics for unique compounds."""
    metadata = dataset.metadata
    if smiles_column not in metadata.columns:
        raise KeyError(f"SMILES column was not found: {smiles_column}")
    levels = [c for c in CLASSIFICATION_COLUMNS if c in metadata.columns]
    compounds = metadata[[smiles_column, *levels]].copy()
    compounds[smiles_column] = compounds[smiles_column].fillna("").astype(str).str.strip()
    compounds = compounds[compounds[smiles_column] != ""].drop_duplicates(smiles_column)

    counts = {int(p.pattern_id): [0, 0] for p in pattern_set.patterns}
    class_counts = Counter()
    class_totals = Counter()
    type_distribution = Counter()
    valid = invalid = 0
    for _, row in tqdm(compounds.iterrows(), total=len(compounds),
                       desc=f"Matching {split_name} cleavage SMARTS", mininterval=1.0):
        try:
            compound = Compound.from_smiles(str(row[smiles_column]))
        except Exception:
            invalid += 1
            continue
        valid += 1
        classes = {c: "" if pd.isna(row[c]) else str(row[c]).strip() for c in levels}
        for c, value in classes.items():
            if value:
                class_totals[c, value] += 1
        matched_types = 0
        for pattern in pattern_set.patterns:
            matches = pattern.matches(compound)
            if not matches:
                continue
            pid = int(pattern.pattern_id)
            matched_types += 1
            counts[pid][0] += 1
            counts[pid][1] += len(matches)
            for c, value in classes.items():
                if value:
                    class_counts[pid, c, value] += 1
        type_distribution[matched_types] += 1

    out = Path(output_dir) / "statistics"
    out.mkdir(parents=True, exist_ok=True)
    patterns = {int(p.pattern_id): p for p in pattern_set.patterns}
    coverage = [{
        "pattern_id": pid, "pattern_name": patterns[pid].name,
        "reactant_smarts": patterns[pid].reactant_smarts,
        "matched_compound_count": values[0], "total_compound_count": valid,
        "compound_coverage": values[0] / valid if valid else 0.0,
        "substructure_match_count": values[1],
    } for pid, values in counts.items()]
    coverage.sort(key=lambda x: (-x["matched_compound_count"], x["pattern_name"], x["pattern_id"]))
    pd.DataFrame(coverage).to_csv(out / f"{split_name}_cleavage_pattern_coverage.tsv", sep="\t", index=False)

    by_class = [{
        "classification_level": level, "classification_value": value,
        "pattern_id": pid, "pattern_name": patterns[pid].name,
        "reactant_smarts": patterns[pid].reactant_smarts,
        "matched_compound_count": class_counts[pid, level, value],
        "total_compound_count": total,
        "compound_coverage": class_counts[pid, level, value] / total,
    } for (level, value), total in class_totals.items() for pid in patterns]
    by_class.sort(key=lambda x: (x["classification_level"], x["classification_value"],
                                 -x["matched_compound_count"], x["pattern_id"]))
    columns = ["classification_level", "classification_value", "pattern_id", "pattern_name",
               "reactant_smarts", "matched_compound_count", "total_compound_count", "compound_coverage"]
    pd.DataFrame(by_class, columns=columns).to_csv(
        out / f"{split_name}_cleavage_pattern_by_class.tsv", sep="\t", index=False)

    distribution = [{"matched_pattern_type_count": n, "compound_count": count,
                     "compound_fraction": count / valid if valid else 0.0}
                    for n, count in sorted(type_distribution.items())]
    pd.DataFrame(distribution).to_csv(
        out / f"{split_name}_matched_pattern_count_distribution.tsv", sep="\t", index=False)
    with (out / f"{split_name}_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"input_record_count": len(dataset), "unique_nonempty_smiles_count": len(compounds),
                   "valid_compound_count": valid, "invalid_smiles_count": invalid,
                   "cleavage_pattern_count": len(pattern_set.patterns),
                   "classification_columns": levels}, f, indent=2)
        f.write("\n")
    print(f"saved cleavage pattern statistics: {out} ({split_name})")
