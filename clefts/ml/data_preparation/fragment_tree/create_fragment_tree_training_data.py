from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from clefts.domain.fragment.cleavage import CleavagePatternSet
from clefts.ml.data_preparation.fragment_tree.assigned_cleavage_event_statistics import (
    write_assigned_cleavage_event_statistics,
)
from clefts.ml.data_preparation.fragment_tree.validation_sampling import (
    sample_validation_dataset,
)
from clefts.libs.mmkit.mmkit import Compound
from clefts.libs.msentity.msentity import MSDataset
from clefts.utils.parallel_subprocess import run_parallel_subprocesses

ORIGINAL_INDEX_COLUMN = "__fragment_tree_original_index"
STRUCTURE_DATA_DIR_NAME = "data"
DEFAULT_PREPROCESSING_CONFIG_NAME = "preprocessing_config.json"
DEFAULT_FRAGMENTER_CONFIG_NAME = "fragmenter.json"
from clefts.ml.input.fragment_tree_training_data import (
    FRAGMENT_TREE_STRUCTURE_GLOBS,
    build_fragment_tree_structure_files,
    build_fragment_tree_structure_files_from_existing,
    group_record_indexes_by_smiles,
    load_fragment_tree_structure_file,
)


def find_structure_files(directory: Path) -> list[Path]:
    return sorted({path for pattern in FRAGMENT_TREE_STRUCTURE_GLOBS for path in directory.glob(pattern)})
from clefts.ml.input.fragment_tree_preprocessing_context import (
    FragmentTreePreprocessingContext,
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build training FragmentTreeStructure files from an MSDataset. "
            "Records are grouped by SMILES, and each SMILES group is saved as one .preft.pt file."
        )
    )
    parser.add_argument(
        "--train-input",
        required=True,
        help="Training input MSDataset path.",
    )
    parser.add_argument(
        "--validation-input",
        default=None,
        help="Optional validation input MSDataset path.",
    )
    parser.add_argument(
        "--validation-smiles-ratio",
        type=float,
        default=0.1,
        help=(
            "Sample this many validation SMILES relative to the number of training "
            "SMILES (for example, 0.1). Sampling is balanced over maximum Tanimoto "
            "similarity to the training set. Default: 0.1."
        ),
    )
    parser.add_argument("--tanimoto-num-bins", type=int, default=10)
    parser.add_argument("--tanimoto-radius", type=int, default=2)
    parser.add_argument("--tanimoto-n-bits", type=int, default=2048)
    parser.add_argument("--validation-sampling-seed", type=int, default=0)
    parser.add_argument(
        "--validation-structures-input-dir",
        default=None,
        help="Optional existing validation structure directory. Mutually exclusive with --validation-input.",
    )
    parser.add_argument(
        "--structure-rebuild-policy",
        choices=("all-fragments", "root", "always"),
        default="all-fragments",
        help=(
            "When using an existing structure directory, rebuild always, rebuild only "
            "when added cleavage patterns match the root compound, or rebuild when "
            "they match any saved fragment."
        ),
    )
    parser.add_argument(
        "--require-precursor-path-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require every supervised peak pathway to explicitly pass through "
            "a precursor node. Use --no-require-precursor-path-targets to keep "
            "pathways without a precursor marker."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Output root directory. Structure .preft.pt files are written under "
            "train_structures/data, and validation .preft.pt files under "
            "validation_structures/data when --validation-input is provided."
        ),
    )
    parser.add_argument(
        "--params",
        required=True,
        help="Fragmenter parameter JSON used for preprocessing.",
    )
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument(
        "--preprocessing-config-output", "--model-config-output",
        dest="preprocessing_config_output",
        default=None,
        help=(
            "Path for immutable preprocessing settings. Defaults to "
            "OUTPUT_DIR/config/preprocessing_config.json."
        ),
    )
    parser.add_argument(
        "--overwrite-preprocessing-config", "--overwrite-model-config",
        dest="overwrite_preprocessing_config",
        action="store_true",
        help="Overwrite an existing copied model config without prompting.",
    )
    parser.add_argument("--smiles-column", default="SMILES")
    parser.add_argument("--precursor-mz-column", default="PrecursorMZ")
    parser.add_argument("--adduct-type-column", default="AdductType")
    parser.add_argument("--collision-energy-column", default="CollisionEnergy")
    parser.add_argument("--instrument-column", default=None)
    parser.add_argument(
        "--max-node",
        type=int,
        default=-1,
        help="Maximum number of nodes when building fragment_ion_tree. Use -1 for no limit.",
    )
    parser.add_argument(
        "--max-edge",
        type=int,
        default=-1,
        help="Maximum number of edges when building fragment_ion_tree. Use -1 for no limit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .preft.pt structure files.",
    )
    parser.add_argument(
        "--save-train-valid-records",
        action="store_true",
        help="Save valid training records as an MSDataset.",
    )
    parser.add_argument(
        "--no-save-validation-valid-records",
        dest="save_validation_valid_records",
        action="store_false",
        help="Do not save valid validation records as an MSDataset.",
    )
    parser.set_defaults(save_validation_valid_records=True)
    parser.add_argument(
        "--train-valid-output",
        default=None,
        help="Output .msds path for valid training records.",
    )
    parser.add_argument(
        "--validation-valid-output",
        default=None,
        help="Output .msds path for valid validation records.",
    )
    parser.add_argument(
        "--train-assignment-score-output",
        default=None,
        help="Output TSV path for training peak assignment scores.",
    )
    parser.add_argument(
        "--validation-assignment-score-output",
        default=None,
        help="Output TSV path for validation peak assignment scores.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel subprocess workers. Use 1 to disable parallel processing.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="Number of SMILES groups per parallel chunk.",
    )
    parser.add_argument(
        "--parallel-temp-dir",
        default=None,
        help="Temporary directory for parallel chunk inputs and outputs.",
    )
    parser.add_argument(
        "--keep-parallel-temp",
        action="store_true",
        help="Keep parallel temporary files after merging.",
    )
    parser.add_argument(
        "--manifest-file",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--structure-output-dir",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--valid-records-output",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--assignment-score-output",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--assigned-cleavage-event-output",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--save-valid-records",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def load_fragmenter_params(params_path: str | Path) -> dict:
    with Path(params_path).open("r", encoding="utf-8") as f:
        params = json.load(f)

    if "probability_model_params" in params:
        return dict(params["probability_model_params"]["fragmenter_params"])
    return params


def load_preprocessing_context(args: argparse.Namespace) -> FragmentTreePreprocessingContext:
    return FragmentTreePreprocessingContext(
        symbols=args.symbols,
        fragmenter_params=load_fragmenter_params(args.params),
        max_node=args.max_node,
        max_edge=args.max_edge,
    )


def _cleavage_pattern_set_from_params_dict(data: dict) -> CleavagePatternSet | None:
    params = data.get("probability_model_params", data)
    fragmenter_params = params.get("fragmenter_params", {})
    builder_params = fragmenter_params.get("fragment_ion_tree_builder", fragmenter_params)
    pattern_set_params = builder_params.get("cleavage_pattern_set")
    if pattern_set_params is None:
        return None
    return CleavagePatternSet.from_dict(pattern_set_params)


def load_cleavage_pattern_set_from_params(path: str | Path) -> CleavagePatternSet | None:
    with Path(path).open("r", encoding="utf-8") as f:
        return _cleavage_pattern_set_from_params_dict(json.load(f))


def resolve_structure_input_data_dir(path: str | Path) -> Path:
    input_dir = Path(path)
    if find_structure_files(input_dir):
        return input_dir
    data_dir = input_dir / STRUCTURE_DATA_DIR_NAME
    if find_structure_files(data_dir):
        return data_dir
    return input_dir


def find_previous_preprocessing_config(structure_input_dir: str | Path) -> Path | None:
    data_dir = resolve_structure_input_data_dir(structure_input_dir)
    candidates = [
        data_dir.parent.parent / "config" / DEFAULT_PREPROCESSING_CONFIG_NAME,
        data_dir.parent / "config" / DEFAULT_PREPROCESSING_CONFIG_NAME,
        data_dir / "config" / DEFAULT_PREPROCESSING_CONFIG_NAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _pattern_matches_smiles(pattern, smiles: str) -> bool:
    try:
        result = pattern.fragment(Compound.from_smiles(smiles))
    except Exception:
        return False
    return result is not None and len(result.products) > 0


def make_structure_rebuild_decider(
    *,
    policy: str,
    current_pattern_set: CleavagePatternSet,
    previous_pattern_set: CleavagePatternSet | None,
):
    if policy == "always":
        return lambda item: True

    if previous_pattern_set is not None:
        previous_id_by_key = {pattern.key: pattern.pattern_id for pattern in previous_pattern_set.patterns}
        for pattern in current_pattern_set.patterns:
            previous_id = previous_id_by_key.get(pattern.key)
            if previous_id is not None and int(previous_id) != int(pattern.pattern_id):
                print(
                    "[WARN] Existing cleavage pattern IDs changed in the current PatternSet; "
                    "all existing structures will be rebuilt.",
                    file=sys.stderr,
                )
                return lambda item: True

    previous_keys = set(previous_pattern_set.identity()) if previous_pattern_set is not None else set()
    added_patterns = [
        pattern
        for pattern in current_pattern_set.patterns
        if pattern.key not in previous_keys
    ]
    if not added_patterns:
        return lambda item: False

    def should_rebuild(item) -> bool:
        metadata = dict(item.metadata)
        root_smiles = str(metadata.get("smiles") or item.structure.node_smiles[0])
        if policy == "root":
            smiles_values = [root_smiles]
        elif policy == "all-fragments":
            smiles_values = [str(smiles) for smiles in item.structure.node_smiles.tolist()]
        else:
            raise ValueError(f"Unknown structure rebuild policy: {policy}")

        for smiles in dict.fromkeys(smiles_values):
            for pattern in added_patterns:
                if _pattern_matches_smiles(pattern, smiles):
                    return True
        return False

    return should_rebuild


def default_preprocessing_config_output(output_root: str | Path) -> Path:
    return Path(output_root) / "config" / DEFAULT_PREPROCESSING_CONFIG_NAME


def write_preprocessing_config(
    *,
    context: FragmentTreePreprocessingContext,
    output_file: str | Path,
    overwrite: bool = False,
) -> None:
    output_path = Path(output_file)

    if output_path.exists() and not overwrite:
        with output_path.open("r", encoding="utf-8") as f:
            existing_config = json.load(f)
        if existing_config == context.to_dict():
            print(f"kept identical existing preprocessing config: {output_path}")
            return
        raise ValueError(
            "Preprocessing settings are immutable once structure data exists. "
            f"The requested settings differ from {output_path}. Use a different "
            "output directory, or rebuild all data with --overwrite and "
            "--overwrite-preprocessing-config."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(context.to_dict(), f, indent=2)
        f.write("\n")
    print(f"wrote preprocessing config: {output_path}")


def write_fragmenter_config(
    *,
    context: FragmentTreePreprocessingContext,
    structure_dir: str | Path,
    overwrite: bool = False,
) -> None:
    """Write a config that can be loaded directly by ``Fragmenter.from_json``."""
    output_path = Path(structure_dir) / DEFAULT_FRAGMENTER_CONFIG_NAME
    config = context.fragmenter.to_dict()

    if output_path.exists() and not overwrite:
        with output_path.open("r", encoding="utf-8") as f:
            existing_config = json.load(f)
        if existing_config == config:
            print(f"kept identical existing fragmenter config: {output_path}")
            return
        raise ValueError(
            f"The fragmenter settings differ from {output_path}. Use a different "
            "output directory, or rebuild all data with --overwrite."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"wrote fragmenter config: {output_path}")


def structure_data_dir(structure_dir: str | Path) -> Path:
    return Path(structure_dir) / STRUCTURE_DATA_DIR_NAME


def save_valid_records(
    *,
    dataset: MSDataset,
    valid_record_indexes: list[int],
    output_file: str | Path,
) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unique_indexes = list(dict.fromkeys(int(index) for index in valid_record_indexes))
    dataset[unique_indexes].save(str(output_path))
    print(f"saved valid records: {len(unique_indexes)} -> {output_path}")


def _metadata_value(row: pd.Series, column: str) -> object:
    if column not in row.index:
        return ""
    value = row[column]
    if pd.isna(value):
        return ""
    return value


def _assigned_peak_indexes_by_sample(structure) -> dict[int, set[int]]:
    sample_to_peak_indexes: dict[int, set[int]] = {}
    if not hasattr(structure, "target_sample_index") or structure.target_sample_index.numel() == 0:
        return sample_to_peak_indexes

    sample_indexes = structure.target_sample_index.detach().cpu().tolist()
    peak_indexes = structure.target_peak_index.detach().cpu().tolist()
    for sample_index, peak_index in zip(sample_indexes, peak_indexes):
        sample_to_peak_indexes.setdefault(int(sample_index), set()).add(int(peak_index))
    return sample_to_peak_indexes


def write_assignment_score_tsv(
    *,
    dataset: MSDataset,
    structure_files: list[Path],
    output_file: str | Path,
    smiles_column: str,
) -> None:
    rows: list[dict[str, object]] = []
    metadata = dataset.metadata

    for structure_file in tqdm(
        structure_files,
        desc="Writing assignment score TSV",
        mininterval=1.0,
    ):
        if not structure_file.exists():
            print(
                f"[WARN] Missing structure file while writing assignment scores: {structure_file}",
                file=sys.stderr,
            )
            continue

        item = load_fragment_tree_structure_file(structure_file, map_location="cpu")
        structure = item.structure
        record_indexes = [int(index) for index in item.metadata.get("record_indexes", [])]
        sample_indexes = [int(index) for index in item.metadata.get("sample_indexes", [])]
        assigned_by_sample = _assigned_peak_indexes_by_sample(structure)

        for local_record_index, sample_index in zip(record_indexes, sample_indexes):
            if int(sample_index) < 0:
                continue
            if local_record_index < 0 or local_record_index >= len(dataset):
                continue

            row = metadata.iloc[local_record_index]
            spectrum = dataset[local_record_index]
            intensities = np.asarray(
                [float(peak.intensity) for peak in spectrum.peaks],
                dtype=np.float64,
            )
            finite_mask = np.isfinite(intensities)
            total_intensity = float(intensities[finite_mask].sum()) if intensities.size else 0.0
            total_peak_count = int(intensities.size)

            assigned_peak_indexes = sorted(
                peak_index
                for peak_index in assigned_by_sample.get(int(sample_index), set())
                if 0 <= int(peak_index) < total_peak_count
            )
            assigned_intensity = float(intensities[assigned_peak_indexes].sum()) if assigned_peak_indexes else 0.0
            assignment_score = assigned_intensity / total_intensity if total_intensity > 0 else 0.0
            source_index = _metadata_value(row, ORIGINAL_INDEX_COLUMN)
            if source_index == "":
                source_index = int(local_record_index)

            rows.append(
                {
                    "index": int(source_index),
                    "SpecID": _metadata_value(row, "SpecID"),
                    "assignment_score": assignment_score,
                    "assigned_intensity": assigned_intensity,
                    "total_intensity": total_intensity,
                    "assigned_peak_count": int(len(assigned_peak_indexes)),
                    "total_peak_count": total_peak_count,
                    "smiles": _metadata_value(row, smiles_column),
                    "structure_file": Path(structure_file).name,
                }
            )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "index",
        "SpecID",
        "assignment_score",
        "assigned_intensity",
        "total_intensity",
        "assigned_peak_count",
        "total_peak_count",
        "smiles",
        "structure_file",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(output_path, sep="\t", index=False)
    print(f"saved assignment scores: {len(rows)} -> {output_path}")


def build_structure_files_for_input(
    *,
    input_path: str,
    output_dir: str | Path,
    args: argparse.Namespace,
    context: FragmentTreePreprocessingContext,
    manifest_file: str | Path | None = None,
    save_valid: bool = False,
    valid_records_output: str | Path | None = None,
    assignment_score_output: str | Path | None = None,
    assigned_cleavage_event_output: str | Path | None = None,
) -> list[Path]:
    dataset = MSDataset.load(input_path)
    print(f"input: {input_path}")
    print(f"output_dir: {output_dir}")

    valid_record_indexes: list[int] = []
    saved_files = build_fragment_tree_structure_files(
        dataset=dataset,
        feature_model=context,
        output_dir=output_dir,
        smiles_column=args.smiles_column,
        precursor_mz_column=args.precursor_mz_column,
        adduct_type_column=args.adduct_type_column,
        collision_energy_column=args.collision_energy_column,
        instrument_column=args.instrument_column,
        max_node=args.max_node,
        max_edge=args.max_edge,
        overwrite=args.overwrite,
        manifest_file=manifest_file,
        valid_record_indexes=valid_record_indexes if save_valid else None,
        require_precursor_path_targets=args.require_precursor_path_targets,
    )

    if save_valid:
        if valid_records_output is None:
            raise ValueError("valid_records_output is required when save_valid=True.")
        save_valid_records(
            dataset=dataset,
            valid_record_indexes=valid_record_indexes,
            output_file=valid_records_output,
        )

    if assignment_score_output is not None:
        write_assignment_score_tsv(
            dataset=dataset,
            structure_files=saved_files,
            output_file=assignment_score_output,
            smiles_column=args.smiles_column,
        )

    if assigned_cleavage_event_output is not None:
        write_assigned_cleavage_event_statistics(
            structure_files=saved_files,
            pattern_set=context.fragmenter.cleavage_pattern_set,
            output_file=assigned_cleavage_event_output,
            num_workers=1,
        )

    print(f"saved structure files: {len(saved_files)}")
    for path in saved_files[:5]:
        print(f"  {path}")

    return saved_files


def build_structure_files_for_existing_input(
    *,
    structures_input_dir: str | Path,
    output_dir: str | Path,
    args: argparse.Namespace,
    context: FragmentTreePreprocessingContext,
    manifest_file: str | Path | None = None,
) -> list[Path]:
    input_data_dir = resolve_structure_input_data_dir(structures_input_dir)
    previous_config = find_previous_preprocessing_config(input_data_dir)
    previous_pattern_set = None
    if previous_config is not None:
        previous_pattern_set = load_cleavage_pattern_set_from_params(previous_config)
        print(f"previous model config: {previous_config}")
    else:
        print(
            "[WARN] Previous model config was not found near the structure input. "
            "All current cleavage patterns will be treated as added patterns.",
            file=sys.stderr,
        )

    current_pattern_set = context.fragmenter.cleavage_pattern_set
    should_rebuild = make_structure_rebuild_decider(
        policy=args.structure_rebuild_policy,
        current_pattern_set=current_pattern_set,
        previous_pattern_set=previous_pattern_set,
    )
    print(f"structures input: {input_data_dir}")
    print(f"output_dir: {output_dir}")
    print(f"structure_rebuild_policy: {args.structure_rebuild_policy}")

    saved_files = build_fragment_tree_structure_files_from_existing(
        input_dir=input_data_dir,
        feature_model=context,
        output_dir=output_dir,
        should_rebuild=should_rebuild,
        max_node=args.max_node,
        max_edge=args.max_edge,
        overwrite=args.overwrite,
        manifest_file=manifest_file,
        require_precursor_path_targets=args.require_precursor_path_targets,
    )

    print(f"saved structure files: {len(saved_files)}")
    for path in saved_files[:5]:
        print(f"  {path}")
    return saved_files


def make_smiles_chunks(
    dataset: MSDataset,
    *,
    smiles_column: str,
    chunk_size: int,
) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    groups = group_record_indexes_by_smiles(dataset, smiles_column=smiles_column)
    items = list(groups.items())
    chunks: list[list[int]] = []
    for start in range(0, len(items), chunk_size):
        chunk_items = items[start : start + chunk_size]
        chunks.append([index for _, indexes in chunk_items for index in indexes])
    return chunks


def merge_tsv_files(input_files: list[Path], output_file: Path) -> None:
    frames = [pd.read_csv(path, sep="\t") for path in input_files if path.exists()]
    if not frames:
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output_file, sep="\t", index=False)


def merge_assigned_cleavage_event_tsv_files(
    input_files: list[Path],
    output_file: Path,
) -> None:
    frames = [pd.read_csv(path, sep="\t") for path in input_files if path.exists()]
    key_columns = [
        "reactant_smarts",
        "matched_substructure",
        "pattern_id",
    ]
    value_columns = [
        "assigned_cleavage_event_count",
        "assigned_pathway_count",
        "sample_count",
    ]
    if frames:
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.groupby(key_columns, as_index=False, dropna=False)[value_columns].sum()
        merged = merged.sort_values(
            ["assigned_cleavage_event_count", "reactant_smarts", "matched_substructure"],
            ascending=[False, True, True],
            kind="stable",
        )
    else:
        merged = pd.DataFrame(columns=[*key_columns, *value_columns])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_columns = [
        "reactant_smarts",
        "matched_substructure",
        "assigned_cleavage_event_count",
        "assigned_pathway_count",
        "sample_count",
        "pattern_id",
    ]
    merged.to_csv(output_file, sep="\t", index=False, columns=output_columns)

    sample_input_files = [
        path.with_name(f"{path.stem}_by_sample{path.suffix}")
        for path in input_files
    ]
    sample_output_file = output_file.with_name(
        f"{output_file.stem}_by_sample{output_file.suffix}"
    )
    merge_tsv_files(sample_input_files, sample_output_file)

    derived_tables = (
        ("by_pattern", ["pattern_id", "reactant_smarts"]),
        (
            "by_pattern_reaction",
            ["pattern_id", "reaction_id", "reactant_smarts"],
        ),
        (
            "by_pattern_reaction_product",
            [
                "pattern_id", "reaction_id", "product_molecule_id",
                "reactant_smarts",
            ],
        ),
    )
    for suffix, derived_key_columns in derived_tables:
        derived_input_files = [
            path.with_name(f"{path.stem}_{suffix}{path.suffix}")
            for path in input_files
        ]
        derived_frames = [
            pd.read_csv(path, sep="\t")
            for path in derived_input_files
            if path.exists()
        ]
        derived_output_file = output_file.with_name(
            f"{output_file.stem}_{suffix}{output_file.suffix}"
        )
        if derived_frames:
            derived = pd.concat(derived_frames, ignore_index=True)
            derived = derived.groupby(
                derived_key_columns, as_index=False, dropna=False
            )[value_columns].sum()
            derived = derived.sort_values(
                ["assigned_cleavage_event_count", *derived_key_columns],
                ascending=[False, *([True] * len(derived_key_columns))],
                kind="stable",
            )
        else:
            derived = pd.DataFrame(columns=[*derived_key_columns, *value_columns])
        derived.to_csv(derived_output_file, sep="\t", index=False)

    print(f"saved merged assigned cleavage events: {output_file}")
    print(f"saved merged per-sample cleavage events: {sample_output_file}")


def merge_msdatasets(input_files: list[Path], output_file: Path, *, empty_like: MSDataset) -> None:
    datasets = [MSDataset.load(str(path)) for path in input_files if path.exists()]
    datasets = [dataset for dataset in datasets if len(dataset) > 0]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if datasets:
        MSDataset.concat(datasets, description=f"Valid records merged for {output_file.name}").save(str(output_file))
    else:
        empty_like[[]].save(str(output_file))
    print(f"saved merged valid records: {output_file}")


def run_parallel_for_input(
    *,
    input_path: str,
    structure_output_dir: Path,
    args: argparse.Namespace,
    manifest_file: Path,
    save_valid: bool,
    valid_records_output: Optional[Path],
    assignment_score_output: Path,
    split_name: str,
) -> None:
    if args.num_workers <= 1:
        raise ValueError("num_workers must be greater than 1 for parallel processing.")

    dataset = MSDataset.load(input_path)
    chunks = make_smiles_chunks(
        dataset,
        smiles_column=args.smiles_column,
        chunk_size=args.chunk_size,
    )
    if not chunks:
        raise RuntimeError(f"No SMILES groups were found for {split_name}.")

    temp_root = parallel_temp_dir(args, Path(args.output_dir), split_name)
    temp_root.mkdir(parents=True, exist_ok=True)

    print(
        f"{split_name}: split into {len(chunks)} chunks "
        f"({args.chunk_size} SMILES groups per chunk)."
    )

    commands: list[list[str]] = []
    part_manifests: list[Path] = []
    part_valid_outputs: list[Path] = []
    part_score_outputs: list[Path] = []
    part_event_outputs: list[Path] = []
    module_name = "clefts.ml.data_preparation.fragment_tree.create_fragment_tree_training_data"

    for chunk_index, record_indexes in enumerate(
        tqdm(chunks, desc=f"Preparing {split_name} chunks", mininterval=1.0)
    ):
        temp_input = temp_root / f"part_{chunk_index:06d}.msds"
        temp_manifest = temp_root / f"part_{chunk_index:06d}_manifest.tsv"
        temp_valid = temp_root / f"part_{chunk_index:06d}_valid.msds"
        temp_score = temp_root / f"part_{chunk_index:06d}_assignment_scores.tsv"
        temp_events = temp_root / f"part_{chunk_index:06d}_assigned_cleavage_events.tsv"
        chunk_dataset = dataset[record_indexes].copy()
        chunk_dataset[ORIGINAL_INDEX_COLUMN] = list(record_indexes)
        chunk_dataset.save(str(temp_input))

        command = [
            sys.executable,
            "-m",
            module_name,
            "--train-input",
            str(temp_input),
            "--output-dir",
            str(args.output_dir),
            "--structure-output-dir",
            str(structure_output_dir),
            "--params",
            str(args.params),
            "--symbols",
            *[str(symbol) for symbol in args.symbols],
            "--smiles-column",
            str(args.smiles_column),
            "--precursor-mz-column",
            str(args.precursor_mz_column),
            "--adduct-type-column",
            str(args.adduct_type_column),
            "--collision-energy-column",
            str(args.collision_energy_column),
            "--max-node",
            str(args.max_node),
            "--max-edge",
            str(args.max_edge),
            "--manifest-file",
            str(temp_manifest),
            "--num-workers",
            "1",
            "--assignment-score-output",
            str(temp_score),
            "--assigned-cleavage-event-output",
            str(temp_events),
        ]
        part_score_outputs.append(temp_score)
        part_event_outputs.append(temp_events)
        if args.instrument_column is not None:
            command.extend(["--instrument-column", str(args.instrument_column)])
        if not args.require_precursor_path_targets:
            command.append("--no-require-precursor-path-targets")
        if save_valid:
            command.extend(
                [
                    "--save-valid-records",
                    "--valid-records-output",
                    str(temp_valid),
                ]
            )
            part_valid_outputs.append(temp_valid)
        commands.append(command)
        part_manifests.append(temp_manifest)

    run_parallel_subprocesses(
        commands_list=commands,
        max_workers=args.num_workers,
        print_output=False,
    )

    merge_tsv_files(part_manifests, manifest_file)
    merge_tsv_files(part_score_outputs, assignment_score_output)
    print(f"saved merged assignment scores: {assignment_score_output}")
    statistics_dir = Path(args.output_dir) / "statistics"
    for legacy_name in (
        f"{split_name}_summary.json",
        f"{split_name}_cleavage_pattern_coverage.tsv",
        f"{split_name}_cleavage_pattern_by_class.tsv",
        f"{split_name}_matched_pattern_count_distribution.tsv",
    ):
        (statistics_dir / legacy_name).unlink(missing_ok=True)
    event_output = statistics_dir / f"{split_name}_assigned_cleavage_events.tsv"
    merge_assigned_cleavage_event_tsv_files(part_event_outputs, event_output)
    if save_valid:
        if valid_records_output is None:
            raise ValueError("valid_records_output is required when save_valid=True.")
        merge_msdatasets(part_valid_outputs, valid_records_output, empty_like=dataset)

    if not args.keep_parallel_temp:
        for path in temp_root.glob("part_*"):
            path.unlink(missing_ok=True)
        try:
            temp_root.rmdir()
        except OSError:
            pass


def default_valid_output(structure_dir: Path) -> Path:
    return structure_dir / "valid_records.msds"


def default_assignment_score_output(structure_dir: Path) -> Path:
    return structure_dir / "assignment_scores.tsv"


def write_split_cleavage_event_statistics(
    *,
    structure_data_directory: Path,
    output_root: Path,
    split_name: str,
    pattern_set: CleavagePatternSet,
    num_workers: int,
) -> None:
    statistics_dir = output_root / "statistics"
    for legacy_name in (
        f"{split_name}_summary.json",
        f"{split_name}_cleavage_pattern_coverage.tsv",
        f"{split_name}_cleavage_pattern_by_class.tsv",
        f"{split_name}_matched_pattern_count_distribution.tsv",
    ):
        (statistics_dir / legacy_name).unlink(missing_ok=True)

    write_assigned_cleavage_event_statistics(
        structure_files=find_structure_files(structure_data_directory),
        pattern_set=pattern_set,
        output_file=(
            output_root
            / "statistics"
            / f"{split_name}_assigned_cleavage_events.tsv"
        ),
        num_workers=num_workers,
    )


def parallel_temp_dir(args: argparse.Namespace, output_root: Path, split_name: str) -> Path:
    if args.parallel_temp_dir is not None:
        return Path(args.parallel_temp_dir) / split_name
    return output_root / f"_{split_name}_parallel_tmp"


def remove_existing_dirs(directories: list[Path], *, description: str) -> None:
    for directory in directories:
        if not directory.exists():
            continue
        if not directory.is_dir() or directory.is_symlink():
            raise NotADirectoryError(f"Expected a {description} directory: {directory}")
        shutil.rmtree(directory)
        print(f"removed existing {description} directory: {directory}")


def prepare_validation_input(
    *,
    args: argparse.Namespace,
    train_structure_data_dir: Path,
    validation_structure_dir: Path,
) -> str | None:
    if args.validation_input is None or args.validation_smiles_ratio is None:
        return args.validation_input
    train_smiles = []
    for structure_file in find_structure_files(train_structure_data_dir):
        item = load_fragment_tree_structure_file(structure_file, map_location="cpu")
        smiles = item.metadata.get("smiles")
        if smiles is None and len(item.structure.node_smiles) > 0:
            smiles = item.structure.node_smiles[0]
        if smiles is not None:
            train_smiles.append(str(smiles))
    if not train_smiles:
        raise RuntimeError(
            f"No completed training structures were found in {train_structure_data_dir}."
        )

    sampled_input = validation_structure_dir / "sampled_input.msds"
    report_file = validation_structure_dir / "max_tanimoto_index.tsv"
    output = sample_validation_dataset(
        train_smiles=train_smiles,
        validation_dataset=MSDataset.load(args.validation_input),
        smiles_column=args.smiles_column,
        ratio=args.validation_smiles_ratio,
        num_bins=args.tanimoto_num_bins,
        radius=args.tanimoto_radius,
        n_bits=args.tanimoto_n_bits,
        seed=args.validation_sampling_seed,
        output_file=sampled_input,
        report_file=report_file,
    )
    sampled_count = len(pd.read_csv(report_file, sep="\t"))
    print(
        f"sampled validation SMILES: {sampled_count} "
        f"(ratio={args.validation_smiles_ratio}) -> {output}"
    )
    print(f"saved validation max Tanimoto indexes: {report_file}")
    return str(output)


def write_split_result_markers(
    *,
    output_root: Path,
    args: argparse.Namespace,
) -> list[Path]:
    """Write one independently openable Workbench result per generated split."""
    written = []
    split_inputs = {
        "train": args.train_input,
        "validation": args.validation_input or args.validation_structures_input_dir,
    }
    for split_name, source_input in split_inputs.items():
        structure_dir = output_root / f"{split_name}_structures"
        manifest_file = structure_dir / "manifest.tsv"
        if not manifest_file.exists():
            continue
        marker = structure_dir / "fragment-tree.pft"
        payload = {
            "schemaVersion": 1,
            "application": "fragment-tree-data-preparation",
            "resultType": "fragment-tree-structure-split",
            "split": split_name,
            "status": "completed",
            "outputDirectory": str(structure_dir.resolve()),
            "sourceInput": None if source_input is None else str(source_input),
            "manifest": "manifest.tsv",
            "dataDirectory": "data",
            "finishedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        written.append(marker)
        print(f"wrote {split_name} result: {marker}")
    return written


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = parse_args()

    generator = None
    output_root = Path(args.output_dir)

    validation_uses_structures = args.validation_structures_input_dir is not None
    if args.validation_input is not None and validation_uses_structures:
        raise ValueError("--validation-input and --validation-structures-input-dir are mutually exclusive.")

    if args.structure_output_dir is not None:
        generator = load_preprocessing_context(args)
        build_structure_files_for_input(
            input_path=args.train_input,
            output_dir=Path(args.structure_output_dir),
            args=args,
            context=generator,
            manifest_file=args.manifest_file,
            save_valid=args.save_valid_records,
            valid_records_output=args.valid_records_output,
            assignment_score_output=args.assignment_score_output,
            assigned_cleavage_event_output=args.assigned_cleavage_event_output,
        )
        return

    train_structure_dir = output_root / "train_structures"
    validation_structure_dir = output_root / "validation_structures"
    train_structure_data_dir = structure_data_dir(train_structure_dir)
    validation_structure_data_dir = structure_data_dir(validation_structure_dir)
    train_manifest_file = (
        Path(args.manifest_file)
        if args.manifest_file is not None
        else train_structure_dir / "manifest.tsv"
    )
    train_valid_output = (
        Path(args.train_valid_output)
        if args.train_valid_output is not None
        else default_valid_output(train_structure_dir)
    )
    validation_valid_output = (
        Path(args.validation_valid_output)
        if args.validation_valid_output is not None
        else default_valid_output(validation_structure_dir)
    )
    train_assignment_score_output = (
        Path(args.train_assignment_score_output)
        if args.train_assignment_score_output is not None
        else default_assignment_score_output(train_structure_dir)
    )
    validation_assignment_score_output = (
        Path(args.validation_assignment_score_output)
        if args.validation_assignment_score_output is not None
        else default_assignment_score_output(validation_structure_dir)
    )

    if args.overwrite:
        structure_dirs = [train_structure_dir]
        parallel_temp_dirs = [parallel_temp_dir(args, output_root, "train")]
        if args.validation_input is not None or validation_uses_structures:
            structure_dirs.append(validation_structure_dir)
            parallel_temp_dirs.append(parallel_temp_dir(args, output_root, "validation"))
        remove_existing_dirs(structure_dirs, description="structure")
        remove_existing_dirs(parallel_temp_dirs, description="parallel temp")

    generator = load_preprocessing_context(args)
    preprocessing_config_output_path = (
        Path(args.preprocessing_config_output)
        if args.preprocessing_config_output is not None
        else default_preprocessing_config_output(output_root)
    )
    if (
        args.overwrite_preprocessing_config
        and train_structure_data_dir.exists()
        and bool(find_structure_files(train_structure_data_dir))
        and not args.overwrite
    ):
        raise ValueError(
            "--overwrite-preprocessing-config cannot be used while retaining existing "
            "training structures. Add --overwrite to rebuild them."
        )
    write_preprocessing_config(
        context=generator,
        output_file=preprocessing_config_output_path,
        overwrite=bool(args.overwrite_preprocessing_config),
    )
    write_fragmenter_config(
        context=generator,
        structure_dir=train_structure_dir,
        overwrite=bool(args.overwrite),
    )
    if args.validation_input is not None or validation_uses_structures:
        write_fragmenter_config(
            context=generator,
            structure_dir=validation_structure_dir,
            overwrite=bool(args.overwrite),
        )

    pattern_set = generator.fragmenter.cleavage_pattern_set

    if validation_uses_structures:
        if args.num_workers > 1:
            run_parallel_for_input(
                input_path=args.train_input,
                structure_output_dir=train_structure_data_dir,
                args=args,
                manifest_file=train_manifest_file,
                save_valid=bool(args.save_train_valid_records),
                valid_records_output=train_valid_output,
                assignment_score_output=train_assignment_score_output,
                split_name="train",
            )
        else:
            build_structure_files_for_input(
                input_path=args.train_input,
                output_dir=train_structure_data_dir,
                args=args,
                context=generator,
                manifest_file=train_manifest_file,
                save_valid=bool(args.save_train_valid_records),
                valid_records_output=train_valid_output,
                assignment_score_output=train_assignment_score_output,
            )

        build_structure_files_for_existing_input(
            structures_input_dir=args.validation_structures_input_dir,
            output_dir=validation_structure_data_dir,
            args=args,
            context=generator,
            manifest_file=validation_structure_dir / "manifest.tsv",
        )
        write_split_cleavage_event_statistics(
            structure_data_directory=train_structure_data_dir,
            output_root=output_root,
            split_name="train",
            pattern_set=pattern_set,
            num_workers=max(1, int(args.num_workers)),
        )
        write_split_cleavage_event_statistics(
            structure_data_directory=validation_structure_data_dir,
            output_root=output_root,
            split_name="validation",
            pattern_set=pattern_set,
            num_workers=max(1, int(args.num_workers)),
        )
        write_split_result_markers(output_root=output_root, args=args)
        return

    if args.num_workers > 1:
        run_parallel_for_input(
            input_path=args.train_input,
            structure_output_dir=train_structure_data_dir,
            args=args,
            manifest_file=train_manifest_file,
            save_valid=bool(args.save_train_valid_records),
            valid_records_output=train_valid_output,
            assignment_score_output=train_assignment_score_output,
            split_name="train",
        )
        validation_input = prepare_validation_input(
            args=args,
            train_structure_data_dir=train_structure_data_dir,
            validation_structure_dir=validation_structure_dir,
        )
        if validation_input is not None:
            run_parallel_for_input(
                input_path=validation_input,
                structure_output_dir=validation_structure_data_dir,
                args=args,
                manifest_file=validation_structure_dir / "manifest.tsv",
                save_valid=bool(args.save_validation_valid_records),
                valid_records_output=validation_valid_output,
                assignment_score_output=validation_assignment_score_output,
                split_name="validation",
            )
        write_split_result_markers(output_root=output_root, args=args)
        return

    build_structure_files_for_input(
        input_path=args.train_input,
        output_dir=train_structure_data_dir,
        args=args,
        context=generator,
        manifest_file=train_manifest_file,
        save_valid=bool(args.save_train_valid_records),
        valid_records_output=train_valid_output,
        assignment_score_output=train_assignment_score_output,
    )

    validation_input = prepare_validation_input(
        args=args,
        train_structure_data_dir=train_structure_data_dir,
        validation_structure_dir=validation_structure_dir,
    )
    if validation_input is not None:
        build_structure_files_for_input(
            input_path=validation_input,
            output_dir=validation_structure_data_dir,
            args=args,
            context=generator,
            manifest_file=validation_structure_dir / "manifest.tsv",
            save_valid=bool(args.save_validation_valid_records),
            valid_records_output=validation_valid_output,
            assignment_score_output=validation_assignment_score_output,
        )


    write_split_cleavage_event_statistics(
        structure_data_directory=train_structure_data_dir,
        output_root=output_root,
        split_name="train",
        pattern_set=pattern_set,
        num_workers=max(1, int(args.num_workers)),
    )
    if args.validation_input is not None:
        write_split_cleavage_event_statistics(
            structure_data_directory=validation_structure_data_dir,
            output_root=output_root,
            split_name="validation",
            pattern_set=pattern_set,
            num_workers=max(1, int(args.num_workers)),
        )
    write_split_result_markers(output_root=output_root, args=args)


if __name__ == "__main__":
    main()
