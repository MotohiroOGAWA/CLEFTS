from __future__ import annotations

import hashlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ...libs.msentity.msentity import MSDataset
from .fragment_tree_structure import FragmentTreeStructure
from .training_fragment_tree_structure import TrainingFragmentTreeStructure
from .training_fragment_tree_structure_builder import TrainingFragmentTreeStructureBuilder

FRAGMENT_TREE_STRUCTURE_SUFFIX = ".preft.pt"
FRAGMENT_TREE_STRUCTURE_GLOBS = ("*.preft.pt", "*.preft", "*.pt")


@dataclass(frozen=True)
class FragmentTreeStructureFileItem:
    """One saved SMILES-group fragment tree structure."""

    path: Path
    structure: FragmentTreeStructure
    metadata: Dict[str, object]


def group_record_indexes_by_smiles(
    dataset: MSDataset,
    *,
    smiles_column: str = "SMILES",
) -> Dict[str, List[int]]:
    """Group visible MSDataset row indexes by SMILES."""

    if smiles_column not in dataset.columns:
        raise KeyError(f"Column not found in MSDataset: {smiles_column}")

    groups: Dict[str, List[int]] = {}
    values = dataset[smiles_column]

    for record_index, value in enumerate(
        tqdm(
            values.tolist(),
            desc="Grouping records by SMILES",
            mininterval=1.0,
        )
    ):
        if pd.isna(value):
            continue
        smiles = str(value)
        groups.setdefault(smiles, []).append(int(record_index))

    return groups


def make_structure_file_stem(smiles: str, *, index: int) -> str:
    """Make a stable readable file stem for one SMILES group."""

    digest = hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:16]
    return f"smiles_{index:06d}_{digest}"


def save_fragment_tree_structure(
    *,
    structure: FragmentTreeStructure,
    output_file: str | Path,
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    """Save one structure and small metadata sidecar into a torch file."""

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "structure": structure.to("cpu"),
            "metadata": {} if metadata is None else dict(metadata),
        },
        output_path,
    )


def load_fragment_tree_structure_file(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    device: Optional[str | torch.device] = None,
) -> FragmentTreeStructureFileItem:
    """Load one saved structure file."""

    file_path = Path(path)
    payload = torch.load(file_path, map_location=map_location)

    if isinstance(payload, FragmentTreeStructure):
        structure = payload
        metadata: Dict[str, object] = {}
    elif isinstance(payload, dict) and "structure" in payload:
        structure = payload["structure"]
        metadata = dict(payload.get("metadata", {}))
    else:
        raise TypeError(
            "Saved fragment-tree structure file must contain either a "
            "FragmentTreeStructure or a dict with key 'structure'. "
            f"Got {type(payload).__name__} from {file_path}."
        )

    if device is not None:
        structure = structure.to(device)

    return FragmentTreeStructureFileItem(
        path=file_path,
        structure=structure,
        metadata=metadata,
    )


class FragmentTreeStructureFileDataset(Dataset[FragmentTreeStructureFileItem]):
    """Torch Dataset for directories of saved SMILES-group structures."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        pattern: str = "*.preft.pt",
        map_location: str | torch.device = "cpu",
        device: Optional[str | torch.device] = None,
        included_samples_by_file: Optional[Dict[str, Set[int]]] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.pattern = pattern
        self.map_location = map_location
        self.device = device
        self.included_samples_by_file = included_samples_by_file
        files = sorted(self.root_dir.glob(pattern))
        self.files = (
            files if included_samples_by_file is None else
            [path for path in files if included_samples_by_file.get(path.name)]
        )

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No fragment-tree structure files matched {pattern!r} in {self.root_dir}."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> FragmentTreeStructureFileItem:
        item = load_fragment_tree_structure_file(
            self.files[index],
            map_location=self.map_location,
            device=self.device,
        )
        if self.included_samples_by_file is None:
            return item
        sample_ids = sorted(self.included_samples_by_file[item.path.name])
        structure = subset_training_structure_samples(item.structure, sample_ids)
        return FragmentTreeStructureFileItem(
            path=item.path, structure=structure, metadata=item.metadata
        )


def subset_training_structure_samples(
    structure: FragmentTreeStructure, sample_ids: Sequence[int]
) -> FragmentTreeStructure:
    """Keep selected samples and their complete CSR target hierarchy."""
    if not isinstance(structure, TrainingFragmentTreeStructure):
        raise TypeError("assignment-score filtering requires TrainingFragmentTreeStructure")
    selected = sorted(set(int(value) for value in sample_ids))
    if not selected or selected[0] < 0 or selected[-1] >= structure.num_samples:
        raise ValueError("sample_ids must select at least one valid sample")
    device = structure.device
    sample_map = torch.full((structure.num_samples,), -1, dtype=torch.long, device=device)
    sample_index = torch.tensor(selected, dtype=torch.long, device=device)
    sample_map[sample_index] = torch.arange(len(selected), device=device)

    sample_edge_mask = torch.isin(structure.sample_edge_index[0].long(), sample_index)
    sample_edge_index = structure.sample_edge_index[:, sample_edge_mask].clone()
    sample_edge_index[0] = sample_map[sample_edge_index[0].long()]
    precursor_mask = torch.isin(structure.precursor_sample_index.long(), sample_index)
    precursor_sample_index = sample_map[structure.precursor_sample_index[precursor_mask].long()]

    peak_ids: List[int] = []
    for sample_id in selected:
        peak_ids.extend(range(int(structure.sample_peak_ptr[sample_id]), int(structure.sample_peak_ptr[sample_id + 1])))
    peak_index = torch.tensor(peak_ids, dtype=torch.long, device=device)
    peak_counts = [int(structure.sample_peak_ptr[i + 1] - structure.sample_peak_ptr[i]) for i in selected]
    sample_peak_ptr = torch.tensor([0, *torch.tensor(peak_counts).cumsum(0).tolist()], dtype=torch.long, device=device)

    formula_ids: List[int] = []
    formula_counts: List[int] = []
    for peak_id in peak_ids:
        start, stop = int(structure.peak_formula_ptr[peak_id]), int(structure.peak_formula_ptr[peak_id + 1])
        formula_ids.extend(range(start, stop)); formula_counts.append(stop - start)
    formula_index = torch.tensor(formula_ids, dtype=torch.long, device=device)
    peak_formula_ptr = torch.tensor([0, *torch.tensor(formula_counts).cumsum(0).tolist()], dtype=torch.long, device=device)

    assignment_ids: List[int] = []
    assignment_counts: List[int] = []
    for formula_id in formula_ids:
        start, stop = int(structure.formula_assignment_ptr[formula_id]), int(structure.formula_assignment_ptr[formula_id + 1])
        assignment_ids.extend(range(start, stop)); assignment_counts.append(stop - start)
    assignment_index = torch.tensor(assignment_ids, dtype=torch.long, device=device)
    formula_assignment_ptr = torch.tensor([0, *torch.tensor(assignment_counts).cumsum(0).tolist()], dtype=torch.long, device=device)

    def subset_csr(data: torch.Tensor, ptr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values: List[torch.Tensor] = []
        counts: List[int] = []
        for assignment_id in assignment_ids:
            start, stop = int(ptr[assignment_id]), int(ptr[assignment_id + 1])
            values.append(data[start:stop]); counts.append(stop - start)
        result = torch.cat(values) if values else data[:0]
        result_ptr = torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()], dtype=torch.long, device=device)
        return result, result_ptr

    expand, expand_ptr = subset_csr(structure.target_expand_node_index, structure.terminal_expand_ptr)
    path, path_ptr = subset_csr(structure.target_path_edge_index, structure.terminal_path_ptr)
    return TrainingFragmentTreeStructure(
        **{name: getattr(structure, name) for name in (
            "node_smiles", "node_graph", "node_graph_offset", "node_formula", "formula_element_order",
            "edge_index", "cleavage_event_edge_index", "cleavage_event", "cleavage_atom_idxs",
            "reactant_tuple_length_table", "product_tuple_length_table", "ion_formula_delta",
            "unsaturation_formula_delta", "radical_formula_delta")},
        tree_sample_ptr=torch.tensor([0, len(selected)], dtype=torch.long, device=device),
        sample_adduct_type_index=structure.sample_adduct_type_index[sample_index],
        sample_ce_value=structure.sample_ce_value[sample_index], sample_edge_index=sample_edge_index,
        precursor_edge_index_path=structure.precursor_edge_index_path[precursor_mask],
        precursor_unsaturation_index=structure.precursor_unsaturation_index[precursor_mask],
        precursor_radical_index=structure.precursor_radical_index[precursor_mask],
        precursor_sample_index=precursor_sample_index,
        sample_peak_mz=structure.sample_peak_mz[peak_index], sample_peak_intensity=structure.sample_peak_intensity[peak_index],
        sample_peak_ptr=sample_peak_ptr, target_formula=structure.target_formula[formula_index],
        peak_formula_ptr=peak_formula_ptr, target_terminal_node_index=structure.target_terminal_node_index[assignment_index],
        target_ion_index=structure.target_ion_index[assignment_index],
        target_unsaturation_index=structure.target_unsaturation_index[assignment_index],
        target_radical_index=structure.target_radical_index[assignment_index],
        formula_assignment_ptr=formula_assignment_ptr, target_expand_node_index=expand,
        terminal_expand_ptr=expand_ptr, target_peak_depth=structure.target_peak_depth[peak_index],
        target_path_edge_index=path, terminal_path_ptr=path_ptr,
    )


def collate_fragment_tree_structure_items(
    items: Sequence[FragmentTreeStructureFileItem],
) -> Dict[str, object]:
    """Collate saved structures into one FragmentTreeStructure batch."""

    if len(items) == 0:
        raise ValueError("items must not be empty.")

    structures = [item.structure for item in items]
    structure_type = type(structures[0])
    structure = structure_type.from_structures(structures)

    return {
        "structure": structure,
        "metadata": [item.metadata for item in items],
        "paths": [str(item.path) for item in items],
    }


def make_fragment_tree_structure_dataloader(
    root_dir: str | Path,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pattern: str = "*.preft.pt",
    map_location: str | torch.device = "cpu",
    device: Optional[str | torch.device] = None,
) -> DataLoader:
    """Create a DataLoader over saved FragmentTreeStructure files."""

    dataset = FragmentTreeStructureFileDataset(
        root_dir,
        pattern=pattern,
        map_location=map_location,
        device=device,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fragment_tree_structure_items,
    )


def build_fragment_tree_structure_files(
    *,
    dataset: MSDataset,
    feature_model,
    output_dir: str | Path,
    smiles_column: str = "SMILES",
    precursor_mz_column: str = "PrecursorMZ",
    adduct_type_column: str = "AdductType",
    collision_energy_column: str = "CollisionEnergy",
    instrument_column: Optional[str] = None,
    max_node: int = -1,
    max_edge: int = -1,
    overwrite: bool = False,
    manifest_file: Optional[str | Path] = None,
    valid_record_indexes: Optional[List[int]] = None,
    require_precursor_path_targets: bool = True,
) -> List[Path]:
    """Build and save one FragmentTreeStructure file per SMILES group."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    groups = group_record_indexes_by_smiles(
        dataset,
        smiles_column=smiles_column,
    )

    if len(groups) == 0:
        raise ValueError("No SMILES groups were found for structure generation.")

    manifest_rows: List[Dict[str, object]] = []
    rejection_rows: List[Dict[str, object]] = []
    saved_files: List[Path] = []
    builder = TrainingFragmentTreeStructureBuilder(feature_model)

    for group_index, (smiles, record_indexes) in enumerate(
        tqdm(groups.items(), desc="Building fragment tree structures", mininterval=1.0)
    ):
        if len(record_indexes) == 0:
            continue

        file_stem = make_structure_file_stem(smiles, index=group_index)
        structure_file = output_path / f"{file_stem}{FRAGMENT_TREE_STRUCTURE_SUFFIX}"

        if structure_file.exists() and not overwrite:
            saved_files.append(structure_file)
            if valid_record_indexes is not None:
                try:
                    payload = torch.load(structure_file, map_location="cpu")
                    metadata = dict(payload.get("metadata", {})) if isinstance(payload, dict) else {}
                    saved_sample_indexes = metadata.get("sample_indexes", [])
                    for record_index, sample_index in zip(record_indexes, saved_sample_indexes):
                        if int(sample_index) >= 0:
                            valid_record_indexes.append(int(record_index))
                except Exception:
                    pass
            manifest_rows.append(
                {
                    "file": structure_file.name,
                    "smiles": smiles,
                    "num_input_records": len(record_indexes),
                    "status": "skipped_exists",
                }
            )
            continue

        sub_dataset = dataset[record_indexes]
        builder.reset()

        try:
            sample_indexes = builder.add_training_sample(
                sub_dataset,
                precursor_mz_column=precursor_mz_column,
                adduct_type_column=adduct_type_column,
                collision_energy_column=collision_energy_column,
                smiles_column=smiles_column,
                instrument_column=instrument_column,
                max_node=max_node,
                max_edge=max_edge,
                require_precursor_path_targets=require_precursor_path_targets,
            )
            valid_sample_count = int((sample_indexes >= 0).sum())
            for local_index, (record_index, sample_index) in enumerate(zip(record_indexes, sample_indexes.tolist())):
                if int(sample_index) >= 0:
                    continue
                metadata_row = dataset.metadata.iloc[record_index]
                rejection = builder.sample_rejection_reasons[local_index] or {
                    "stage": "unknown", "reason": "Sample was rejected without a recorded reason."
                }
                rejection_rows.append({
                    "record_index": int(record_index),
                    "source_record_index": int(metadata_row.get("__fragment_tree_original_index", record_index)),
                    "SpecID": "" if pd.isna(metadata_row.get("SpecID")) else str(metadata_row.get("SpecID")),
                    "smiles": smiles,
                    "adduct": "" if pd.isna(metadata_row.get(adduct_type_column)) else str(metadata_row.get(adduct_type_column)),
                    "collision_energy": "" if pd.isna(metadata_row.get(collision_energy_column)) else str(metadata_row.get(collision_energy_column)),
                    "stage": rejection["stage"],
                    "reason": rejection["reason"],
                })

            if valid_sample_count <= 0:
                manifest_rows.append(
                    {
                        "file": structure_file.name,
                        "smiles": smiles,
                        "num_input_records": len(record_indexes),
                        "rejected_sample_count": len(record_indexes),
                        "rejection_log": "rejected_samples.tsv",
                        "status": "no_valid_samples",
                    }
                )
                continue

            structure = builder.to_structure()
            raw_collision_energy = [None] * int(structure.num_samples)
            raw_adduct_types = [None] * int(structure.num_samples)
            for local_record_index, built_sample_index in enumerate(sample_indexes.tolist()):
                if int(built_sample_index) < 0:
                    continue
                record = sub_dataset[local_record_index]
                raw_collision_energy[int(built_sample_index)] = str(record[collision_energy_column])
                raw_adduct_types[int(built_sample_index)] = str(record[adduct_type_column])
            source_record_indexes = [
                int(dataset.metadata.iloc[index].get("__fragment_tree_original_index", index))
                for index in record_indexes
            ]
            spec_ids = (
                [
                    "" if pd.isna(dataset.metadata.iloc[index]["SpecID"])
                    else str(dataset.metadata.iloc[index]["SpecID"])
                    for index in record_indexes
                ]
                if "SpecID" in dataset.metadata.columns
                else None
            )
            metadata = {
                "target_schema_version": 2,
                "target_depth_policy": "minimum-post-precursor-cleavage-depth",
                "smiles": smiles,
                "record_indexes": [int(index) for index in record_indexes],
                "source_record_indexes": source_record_indexes,
                "sample_indexes": [int(index) for index in sample_indexes.tolist()],
                "num_input_records": int(len(record_indexes)),
                "num_valid_samples": int(structure.num_samples),
                "num_nodes": int(structure.num_nodes),
                "num_edges": int(structure.num_edges),
                "max_node": int(max_node),
                "max_edge": int(max_edge),
                "require_precursor_path_targets": bool(require_precursor_path_targets),
                "sample_collision_energy_raw": raw_collision_energy,
                "sample_adduct_types": raw_adduct_types,
            }
            if spec_ids is not None:
                metadata["spec_ids"] = spec_ids
            save_fragment_tree_structure(
                structure=structure,
                output_file=structure_file,
                metadata=metadata,
            )
            if structure_file.exists():
                saved_files.append(structure_file)
            else:
                raise FileNotFoundError(
                    f"Structure file was not created: {structure_file}"
                )
            if valid_record_indexes is not None:
                for record_index, sample_index in zip(record_indexes, sample_indexes.tolist()):
                    if int(sample_index) >= 0:
                        valid_record_indexes.append(int(record_index))
            manifest_rows.append({"file": structure_file.name, "status": "ok", **metadata})
        except Exception as exc:
            print(
                f"[WARN] Failed to build structure for smiles={smiles!r}: {exc}",
                file=sys.stderr,
            )
            manifest_rows.append(
                {
                    "file": structure_file.name,
                    "smiles": smiles,
                    "num_input_records": len(record_indexes),
                    "status": "error",
                    "message": str(exc),
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = Path(manifest_file) if manifest_file is not None else output_path / "manifest.tsv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, sep="\t", index=False)
    rejection_columns = ["record_index", "source_record_index", "SpecID", "smiles", "adduct", "collision_energy", "stage", "reason"]
    pd.DataFrame(rejection_rows, columns=rejection_columns).to_csv(
        manifest_path.with_name("rejected_samples.tsv"), sep="\t", index=False
    )

    return saved_files


def build_fragment_tree_structure_files_from_existing(
    *,
    input_dir: str | Path,
    feature_model,
    output_dir: str | Path,
    should_rebuild,
    max_node: int = -1,
    max_edge: int = -1,
    overwrite: bool = False,
    manifest_file: Optional[str | Path] = None,
    require_precursor_path_targets: bool = True,
) -> List[Path]:
    """Build/copy structure files from an existing structure directory."""

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    source_files = sorted({path for pattern in FRAGMENT_TREE_STRUCTURE_GLOBS for path in input_path.glob(pattern)})
    if len(source_files) == 0:
        raise FileNotFoundError(f"No structure .preft.pt files found in {input_path}.")

    manifest_rows: List[Dict[str, object]] = []
    rejection_rows: List[Dict[str, object]] = []
    saved_files: List[Path] = []
    builder = TrainingFragmentTreeStructureBuilder(feature_model)

    for source_file in tqdm(source_files, desc="Building from existing structures", mininterval=1.0):
        source_stem = source_file.name
        for suffix in FRAGMENT_TREE_STRUCTURE_GLOBS:
            literal_suffix = suffix.removeprefix("*")
            if source_stem.endswith(literal_suffix):
                source_stem = source_stem[: -len(literal_suffix)]
                break
        target_file = output_path / f"{source_stem}{FRAGMENT_TREE_STRUCTURE_SUFFIX}"
        if target_file.exists() and not overwrite:
            item = load_fragment_tree_structure_file(target_file, map_location="cpu")
            metadata = dict(item.metadata)
            saved_files.append(target_file)
            manifest_rows.append({"file": target_file.name, "smiles": metadata.get("smiles", ""), "num_input_records": metadata.get("num_input_records", ""), "num_valid_samples": metadata.get("num_valid_samples", item.structure.num_samples), "num_nodes": item.structure.num_nodes, "num_edges": item.structure.num_edges, "status": "skipped_exists"})
            continue

        try:
            item = load_fragment_tree_structure_file(source_file, map_location="cpu")
            metadata = dict(item.metadata)
            smiles = str(metadata.get("smiles") or item.structure.node_smiles[0])
            rebuild = bool(should_rebuild(item))

            if not rebuild:
                if source_file.resolve() != target_file.resolve():
                    shutil.copy2(source_file, target_file)
                saved_files.append(target_file)
                manifest_rows.append({"file": target_file.name, "smiles": smiles, "num_input_records": metadata.get("num_input_records", item.structure.num_samples), "num_valid_samples": metadata.get("num_valid_samples", item.structure.num_samples), "num_nodes": item.structure.num_nodes, "num_edges": item.structure.num_edges, "status": "reused_no_new_pattern_match"})
                continue

            builder.reset()
            sample_indexes = builder.add_training_samples_from_structure(item.structure, smiles=smiles, max_node=max_node, max_edge=max_edge, require_precursor_path_targets=require_precursor_path_targets)
            valid_sample_count = int((sample_indexes >= 0).sum())
            raw_adducts = metadata.get("sample_adduct_types", [])
            raw_ce = metadata.get("sample_collision_energy_raw", [])
            spec_ids = metadata.get("spec_ids", [])
            source_indexes = metadata.get("source_record_indexes", metadata.get("record_indexes", []))
            for sample_id, sample_index in enumerate(sample_indexes.tolist()):
                if int(sample_index) >= 0:
                    continue
                rejection = builder.sample_rejection_reasons[sample_id] or {
                    "stage": "unknown", "reason": "Sample was rejected without a recorded reason."
                }
                rejection_rows.append({
                    "record_index": sample_id,
                    "source_record_index": source_indexes[sample_id] if sample_id < len(source_indexes) else sample_id,
                    "SpecID": spec_ids[sample_id] if sample_id < len(spec_ids) else "",
                    "smiles": smiles,
                    "adduct": raw_adducts[sample_id] if sample_id < len(raw_adducts) else str(item.structure.sample_adduct_type_index[sample_id].item()),
                    "collision_energy": raw_ce[sample_id] if sample_id < len(raw_ce) else str(item.structure.sample_ce_value[sample_id].item()),
                    "stage": rejection["stage"],
                    "reason": rejection["reason"],
                })
            if valid_sample_count <= 0:
                manifest_rows.append({"file": target_file.name, "smiles": smiles, "num_input_records": item.structure.num_samples, "rejected_sample_count": item.structure.num_samples, "rejection_log": "rejected_samples.tsv", "status": "no_valid_samples"})
                continue

            record_indexes = metadata.get("record_indexes", [])
            old_sample_indexes = metadata.get("sample_indexes", [])
            if record_indexes and old_sample_indexes:
                valid_record_indexes = [int(record_index) for record_index, old_sample_index in zip(record_indexes, old_sample_indexes) if int(old_sample_index) >= 0]
            else:
                valid_record_indexes = list(range(item.structure.num_samples))

            structure = builder.to_structure()
            new_metadata = {**metadata, "target_schema_version": 2, "target_depth_policy": "minimum-post-precursor-cleavage-depth", "require_precursor_path_targets": bool(require_precursor_path_targets), "smiles": smiles, "record_indexes": valid_record_indexes, "sample_indexes": [int(index) for index in sample_indexes.tolist()], "num_input_records": int(len(valid_record_indexes)), "num_valid_samples": int(structure.num_samples), "num_nodes": int(structure.num_nodes), "num_edges": int(structure.num_edges), "max_node": int(max_node), "max_edge": int(max_edge), "source_structure_file": str(source_file)}
            save_fragment_tree_structure(structure=structure, output_file=target_file, metadata=new_metadata)
            saved_files.append(target_file)
            manifest_rows.append({"file": target_file.name, "status": "rebuilt", **new_metadata})
        except Exception as exc:
            print(f"[WARN] Failed to build structure from {source_file}: {exc}", file=sys.stderr)
            manifest_rows.append({"file": target_file.name, "status": "error", "message": str(exc)})

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = Path(manifest_file) if manifest_file is not None else output_path / "manifest.tsv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, sep="\t", index=False)
    rejection_columns = ["record_index", "source_record_index", "SpecID", "smiles", "adduct", "collision_energy", "stage", "reason"]
    pd.DataFrame(rejection_rows, columns=rejection_columns).to_csv(
        manifest_path.with_name("rejected_samples.tsv"), sep="\t", index=False
    )

    return saved_files
