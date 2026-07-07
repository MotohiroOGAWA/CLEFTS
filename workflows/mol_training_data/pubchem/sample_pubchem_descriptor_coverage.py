from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from rdkit import RDLogger
from tqdm import tqdm

APP_ROOT = Path(__file__).resolve().parents[3]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from clefts.libs.mmkit.mmkit import Compound  # noqa: E402
from clefts.ml.training.mol_training.dataset import DEFAULT_DESCRIPTOR_NAMES, compute_descriptors  # noqa: E402
from clefts.ml.training.mol_training.descriptor_coverage import (  # noqa: E402
    DEFAULT_DESCRIPTOR_BIN_SPECS_BY_NAME,
    DescriptorBinSpec,
    descriptor_bin_bounds,
    descriptor_bin_index,
    descriptor_bin_label,
)
from clefts.utils.parallel_subprocess import run_parallel_subprocesses  # noqa: E402

DEFAULT_INPUT = Path("/workspaces/CLEFTS/raw_data/PubChem/pubchem_cid_smiles.parquet")
DEFAULT_OUTPUT_DIR = Path("/workspaces/CLEFTS/mnt/app/data/processed/pubchem/descriptor_coverage_sample")
DEFAULT_OUTPUT_NAME = "pubchem_descriptor_coverage_sample.parquet"
DEFAULT_REPORT_NAME = "pubchem_descriptor_coverage_sample_report.json"
CID_OUTPUT = "CID"
SMILES_OUTPUT = "SMILES"


def configure_torch_threads(thread_count: int | None) -> None:
    if thread_count is None or thread_count <= 0:
        return
    torch.set_num_threads(thread_count)
    try:
        torch.set_num_interop_threads(thread_count)
    except RuntimeError:
        pass


def subprocess_env(*, worker_threads: int) -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        env[name] = str(max(worker_threads, 1))
    return env


def parse_csv_strings(value: str | Sequence[str]) -> Tuple[str, ...]:
    if isinstance(value, str):
        raw = value.split(",")
    else:
        raw = []
        for item in value:
            raw.extend(str(item).split(","))
    out = tuple(item.strip() for item in raw if item.strip())
    if not out:
        raise ValueError("At least one descriptor name is required.")
    return out


def default_smiles_output_path(output_path: Path) -> Path:
    return output_path.with_suffix(".smiles.txt")


def safe_label(value: object) -> str:
    label = str(value).strip()
    replacements = ((" ", "_"), ("/", "_"), ("\\", "_"), (",", "_"), (":", "_"), ("-", "minus_"), ("+", "plus_"), (".", "p"))
    for old, new in replacements:
        label = label.replace(old, new)
    return label


def descriptor_flag_column(name: str, label: str) -> str:
    return f"descriptor_{safe_label(name)}_{safe_label(label)}"


def iter_randomized_batches(
    input_path: Path,
    cid_column: str,
    smiles_column: str,
    batch_size: int,
    rng: random.Random,
    max_rows: int,
) -> Iterable[pd.DataFrame]:
    parquet_file = pq.ParquetFile(input_path)
    row_groups = list(range(parquet_file.metadata.num_row_groups))
    rng.shuffle(row_groups)
    scanned = 0
    for row_group in tqdm(row_groups, desc="row groups", unit="group"):
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            row_groups=[row_group],
            columns=[cid_column, smiles_column],
        ):
            frame = batch.to_pandas()
            if frame.empty:
                continue
            frame = frame.sample(frac=1.0, random_state=rng.randrange(0, 2**32 - 1)).reset_index(drop=True)
            remaining = max_rows - scanned
            if remaining <= 0:
                return
            frame = frame.iloc[:remaining]
            scanned += len(frame)
            yield frame.rename(columns={cid_column: CID_OUTPUT, smiles_column: SMILES_OUTPUT})


def compute_descriptor_row(smiles: object, descriptor_names: Sequence[str]) -> List[float] | None:
    if pd.isna(smiles):
        return None
    try:
        compound = Compound.from_smiles(str(smiles))
        return compute_descriptors(compound.mol, descriptor_names).detach().cpu().tolist()
    except Exception:
        return None


def write_worker_input(path: Path, frame: pd.DataFrame, *, result_offset: int) -> int:
    result_indices = list(range(result_offset, result_offset + len(frame)))
    table = pa.Table.from_arrays(
        [
            pa.array(result_indices, type=pa.int64()),
            pa.array(frame[CID_OUTPUT].tolist(), type=pa.int64()),
            pa.array(frame[SMILES_OUTPUT].astype(str).tolist(), type=pa.string()),
        ],
        names=["result_index", CID_OUTPUT, SMILES_OUTPUT],
    )
    pq.write_table(table, path, compression="zstd")
    return result_offset + len(frame)


def worker_descriptor_candidates(
    frame: pd.DataFrame,
    *,
    descriptor_names: Sequence[str],
    show_progress: bool = True,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    descriptor_rows: List[List[float]] = []
    invalid_smiles = 0
    processed_rows = 0

    for row in tqdm(
        frame.itertuples(index=False),
        total=len(frame),
        desc="worker descriptors",
        unit="compound",
        leave=False,
        mininterval=1.0,
        disable=not show_progress,
    ):
        processed_rows += 1
        values = compute_descriptor_row(getattr(row, SMILES_OUTPUT), descriptor_names)
        if values is None:
            invalid_smiles += 1
            continue
        rows.append(
            {
                "result_index": int(getattr(row, "result_index")),
                CID_OUTPUT: getattr(row, CID_OUTPUT),
                SMILES_OUTPUT: getattr(row, SMILES_OUTPUT),
            }
        )
        descriptor_rows.append(values)
    return {
        "rows": rows,
        "descriptor_rows": descriptor_rows,
        "processed_rows": processed_rows,
        "invalid_smiles": invalid_smiles,
    }


def run_worker(
    *,
    input_path: Path,
    output_path: Path,
    descriptor_names: Sequence[str],
    worker_threads: int,
) -> None:
    RDLogger.DisableLog("rdApp.*")
    configure_torch_threads(worker_threads)
    table = pq.read_table(input_path)
    frame = table.to_pandas()
    payload = worker_descriptor_candidates(
        frame,
        descriptor_names=descriptor_names,
        show_progress=False,
    )
    with open(output_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_worker_wave(
    *,
    frames: Sequence[pd.DataFrame],
    wave_index: int,
    worker_count: int,
    descriptor_names: Sequence[str],
    work_dir: Path,
    result_offset: int,
    worker_threads: int,
) -> Tuple[List[Dict[str, object]], List[List[float]], Dict[str, object], int]:
    if worker_count == 1:
        rows: List[Dict[str, object]] = []
        descriptor_rows: List[List[float]] = []
        processed_rows = 0
        invalid_smiles = 0
        next_result_offset = result_offset
        for frame in frames:
            worker_frame = frame.rename(columns={"cid": CID_OUTPUT, "smiles": SMILES_OUTPUT}).copy()
            worker_frame.insert(
                0,
                "result_index",
                range(next_result_offset, next_result_offset + len(worker_frame)),
            )
            next_result_offset += len(worker_frame)
            payload = worker_descriptor_candidates(
                worker_frame,
                descriptor_names=descriptor_names,
                show_progress=False,
            )
            rows.extend(payload["rows"])
            descriptor_rows.extend(payload["descriptor_rows"])
            processed_rows += int(payload["processed_rows"])
            invalid_smiles += int(payload["invalid_smiles"])
        paired = sorted(zip(rows, descriptor_rows), key=lambda item: int(item[0]["result_index"]))
        return (
            [row for row, _ in paired],
            [descriptor_row for _, descriptor_row in paired],
            {"processed_rows": processed_rows, "invalid_smiles": invalid_smiles},
            next_result_offset,
        )

    wave_dir = work_dir / f"wave_{wave_index:06d}"
    wave_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    commands_list: List[List[str]] = []
    worker_outputs: List[Path] = []
    next_result_offset = result_offset

    combined_frame = pd.concat(
        [frame.rename(columns={"cid": CID_OUTPUT, "smiles": SMILES_OUTPUT}) for frame in frames],
        ignore_index=True,
    )
    active_workers = max(1, min(worker_count, len(combined_frame)))
    chunk_size = math.ceil(len(combined_frame) / active_workers)

    try:
        for worker_index, chunk_start in enumerate(range(0, len(combined_frame), chunk_size)):
            worker_input = wave_dir / f"worker_{worker_index:03d}.parquet"
            worker_output = wave_dir / f"worker_{worker_index:03d}.pkl"
            worker_frame = combined_frame.iloc[chunk_start : chunk_start + chunk_size].reset_index(drop=True)
            next_result_offset = write_worker_input(worker_input, worker_frame, result_offset=next_result_offset)
            commands_list.append(
                [
                    sys.executable,
                    str(script_path),
                    "--worker-input",
                    str(worker_input),
                    "--worker-output",
                    str(worker_output),
                    "--descriptor-names",
                    ",".join(descriptor_names),
                    "--worker-threads",
                    str(worker_threads),
                ]
            )
            worker_outputs.append(worker_output)

        run_parallel_subprocesses(
            commands_list,
            max_workers=active_workers,
            print_output=False,
            env=subprocess_env(worker_threads=worker_threads),
            desc=f"wave {wave_index:06d} descriptor workers",
        )

        rows: List[Dict[str, object]] = []
        descriptor_rows: List[List[float]] = []
        processed_rows = 0
        invalid_smiles = 0
        for worker_output in worker_outputs:
            with open(worker_output, "rb") as f:
                payload = pickle.load(f)
            rows.extend(payload["rows"])
            descriptor_rows.extend(payload["descriptor_rows"])
            processed_rows += int(payload["processed_rows"])
            invalid_smiles += int(payload["invalid_smiles"])
        paired = sorted(zip(rows, descriptor_rows), key=lambda item: int(item[0]["result_index"]))
        return (
            [row for row, _ in paired],
            [descriptor_row for _, descriptor_row in paired],
            {"processed_rows": processed_rows, "invalid_smiles": invalid_smiles},
            next_result_offset,
        )
    finally:
        shutil.rmtree(wave_dir, ignore_errors=True)


def choose_centered_active_bins(
    values_by_name: Dict[str, List[float]],
    specs_by_name: Dict[str, DescriptorBinSpec],
    *,
    min_count: int,
    center: str,
) -> Dict[str, Dict[str, object]]:
    active: Dict[str, Dict[str, object]] = {}
    for name, values in values_by_name.items():
        spec = specs_by_name[name]
        counts = [0 for _ in range(spec.bin_count)]
        finite_values = [float(value) for value in values if math.isfinite(float(value))]
        for value in finite_values:
            bin_index = descriptor_bin_index(value, spec)
            if bin_index is not None:
                counts[bin_index] += 1
        if not finite_values:
            continue
        center_value = sorted(finite_values)[len(finite_values) // 2] if center == "median" else sum(finite_values) / len(finite_values)
        center_index = descriptor_bin_index(center_value, spec)
        eligible = [idx for idx, count in enumerate(counts) if count >= min_count]
        if not eligible:
            active[name] = {"center": center_value, "bins": [], "counts": counts}
            continue
        if center_index is None or counts[center_index] < min_count:
            center_index = min(eligible, key=lambda idx: abs(idx - (center_index if center_index is not None else 0)))
        left = center_index
        while left - 1 >= 0 and counts[left - 1] >= min_count:
            left -= 1
        right = center_index
        while right + 1 < len(counts) and counts[right + 1] >= min_count:
            right += 1
        bins = []
        for bin_index in range(left, right + 1):
            label = descriptor_bin_label(bin_index)
            lower, upper = descriptor_bin_bounds(spec, bin_index)
            bins.append(
                {
                    "label": label,
                    "bin_index": int(bin_index),
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": int(counts[bin_index]),
                    "column": descriptor_flag_column(name, label),
                }
            )
        active[name] = {"center": center_value, "bins": bins, "counts": counts}
    return active


def row_bin_labels(row: Sequence[float], descriptor_names: Sequence[str], specs_by_name: Dict[str, DescriptorBinSpec]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for idx, name in enumerate(descriptor_names):
        bin_index = descriptor_bin_index(float(row[idx]), specs_by_name[name])
        if bin_index is not None:
            labels[name] = descriptor_bin_label(bin_index)
    return labels


def write_smiles_file(frame: pd.DataFrame, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for smiles in frame[SMILES_OUTPUT].astype(str).tolist():
            if smiles.strip():
                f.write(f"{smiles}\n")


def sample_pubchem_descriptor_coverage(
    *,
    input_path: Path,
    output_path: Path,
    smiles_output_path: Path,
    report_path: Path,
    descriptor_names: Sequence[str],
    min_count: int,
    max_count: int | None,
    discovery_rows: int,
    batch_size: int,
    workers: int,
    worker_threads: int,
    seed: int,
    center: str,
    cid_column: str,
    smiles_column: str,
) -> Dict[str, object]:
    RDLogger.DisableLog("rdApp.*")
    specs_by_name = {name: DEFAULT_DESCRIPTOR_BIN_SPECS_BY_NAME[name] for name in descriptor_names}
    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []
    descriptor_rows: List[List[float]] = []
    invalid_smiles = 0
    processed_rows = 0
    workers = max(int(workers), 1)
    worker_threads = max(int(worker_threads), 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_temp_dir = tempfile.TemporaryDirectory(prefix="descriptor_coverage_workers_", dir=str(output_path.parent))
    work_dir = Path(work_temp_dir.name)

    progress = tqdm(total=discovery_rows, desc="descriptor discovery", unit="record")
    try:
        pending_frames: List[pd.DataFrame] = []
        wave_index = 0
        result_offset = 0
        for frame in iter_randomized_batches(input_path, cid_column, smiles_column, batch_size, rng, discovery_rows):
            pending_frames.append(frame)
            if len(pending_frames) < workers:
                continue
            wave_rows, wave_descriptor_rows, stats, result_offset = run_worker_wave(
                frames=pending_frames,
                wave_index=wave_index,
                worker_count=workers,
                descriptor_names=descriptor_names,
                work_dir=work_dir,
                result_offset=result_offset,
                worker_threads=worker_threads,
            )
            pending_frames = []
            wave_index += 1
            rows.extend({CID_OUTPUT: row[CID_OUTPUT], SMILES_OUTPUT: row[SMILES_OUTPUT]} for row in wave_rows)
            descriptor_rows.extend(wave_descriptor_rows)
            processed_rows += int(stats["processed_rows"])
            invalid_smiles += int(stats["invalid_smiles"])
            progress.update(int(stats["processed_rows"]))

        if pending_frames:
            wave_rows, wave_descriptor_rows, stats, result_offset = run_worker_wave(
                frames=pending_frames,
                wave_index=wave_index,
                worker_count=workers,
                descriptor_names=descriptor_names,
                work_dir=work_dir,
                result_offset=result_offset,
                worker_threads=worker_threads,
            )
            rows.extend({CID_OUTPUT: row[CID_OUTPUT], SMILES_OUTPUT: row[SMILES_OUTPUT]} for row in wave_rows)
            descriptor_rows.extend(wave_descriptor_rows)
            processed_rows += int(stats["processed_rows"])
            invalid_smiles += int(stats["invalid_smiles"])
            progress.update(int(stats["processed_rows"]))
    finally:
        progress.close()
        work_temp_dir.cleanup()

    values_by_name = {
        name: [descriptor_row[idx] for descriptor_row in descriptor_rows]
        for idx, name in enumerate(descriptor_names)
    }
    active = choose_centered_active_bins(
        values_by_name,
        specs_by_name,
        min_count=min_count,
        center=center,
    )
    active_columns = [bin_info["column"] for item in active.values() for bin_info in item["bins"]]
    active_by_name = {
        name: {str(bin_info["label"]) for bin_info in item["bins"]}
        for name, item in active.items()
    }
    if max_count is not None and int(max_count) < int(min_count):
        raise ValueError("max_count must be greater than or equal to min_count when specified.")

    candidate_indices_by_column: Dict[str, List[int]] = {column: [] for column in active_columns}
    matching_columns_by_index: List[List[str]] = []
    for row_index, descriptor_row in enumerate(descriptor_rows):
        labels = row_bin_labels(descriptor_row, descriptor_names, specs_by_name)
        matching_columns = [
            descriptor_flag_column(name, label)
            for name, label in labels.items()
            if label in active_by_name.get(name, set())
        ]
        matching_columns_by_index.append(matching_columns)
        for column in matching_columns:
            if column in candidate_indices_by_column:
                candidate_indices_by_column[column].append(row_index)

    selected_indices_by_column: Dict[str, List[int]] = {}
    selected_index_set: set[int] = set()
    for column, candidate_indices in candidate_indices_by_column.items():
        unique_candidate_indices: List[int] = []
        seen_candidate_smiles: set[str] = set()
        for row_index in candidate_indices:
            smiles = str(rows[row_index][SMILES_OUTPUT]).strip()
            if not smiles or smiles in seen_candidate_smiles:
                continue
            seen_candidate_smiles.add(smiles)
            unique_candidate_indices.append(row_index)
        candidate_indices_by_column[column] = unique_candidate_indices
        candidate_indices = list(unique_candidate_indices)
        if max_count is not None and len(candidate_indices) > int(max_count):
            candidate_indices = sorted(rng.sample(candidate_indices, int(max_count)))
        selected_indices_by_column[column] = candidate_indices
        selected_index_set.update(candidate_indices)

    selected_rows: List[Dict[str, object]] = []
    selected_smiles: set[str] = set()
    for row_index in sorted(selected_index_set):
        base_row = rows[row_index]
        smiles = str(base_row[SMILES_OUTPUT]).strip()
        if not smiles or smiles in selected_smiles:
            continue
        selected_smiles.add(smiles)
        descriptor_row = descriptor_rows[row_index]
        matching_columns = set(matching_columns_by_index[row_index])
        out_row = dict(base_row)
        for idx, name in enumerate(descriptor_names):
            out_row[f"descriptor_{name}"] = float(descriptor_row[idx])
        for column in active_columns:
            out_row[column] = column in matching_columns
        selected_rows.append(out_row)

    selected_bin_counts = {column: len(indices) for column, indices in selected_indices_by_column.items()}
    counts = {column: 0 for column in active_columns}
    for row in selected_rows:
        for column in active_columns:
            if bool(row.get(column, False)):
                counts[column] += 1

    def complete() -> bool:
        return bool(selected_bin_counts) and all(count >= min_count for count in selected_bin_counts.values())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    smiles_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [CID_OUTPUT, SMILES_OUTPUT, *[f"descriptor_{name}" for name in descriptor_names], *active_columns]
    frame = pd.DataFrame(selected_rows)
    if frame.empty:
        frame = pd.DataFrame(columns=columns)
    for column in active_columns:
        if column not in frame.columns:
            frame[column] = False
        frame[column] = frame[column].astype(bool)
    frame = frame.reindex(columns=columns)
    frame.to_parquet(output_path, index=False, compression="zstd")
    write_smiles_file(frame, smiles_output_path)

    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "smiles_output_path": str(smiles_output_path),
        "descriptor_names": list(descriptor_names),
        "min_count": int(min_count),
        "max_count": int(max_count) if max_count is not None else None,
        "discovery_rows": int(discovery_rows),
        "valid_discovery_rows": len(descriptor_rows),
        "processed_rows": int(processed_rows),
        "invalid_smiles": int(invalid_smiles),
        "workers": int(workers),
        "worker_threads": int(worker_threads),
        "center": center,
        "seed": int(seed),
        "batch_size": int(batch_size),
        "selected_rows": len(frame),
        "unique_selected_smiles": len(selected_smiles),
        "complete": complete(),
        "active_bin_candidate_counts": {column: len(indices) for column, indices in candidate_indices_by_column.items()},
        "selected_bin_counts": selected_bin_counts,
        "active_bin_counts": counts,
        "active_bins": active,
        "descriptor_bin_specs": [
            {
                "name": spec.name,
                "lower": float(spec.lower),
                "upper": float(spec.upper),
                "width": float(spec.width),
                "bin_count": int(spec.bin_count),
            }
            for spec in specs_by_name.values()
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def output_paths_from_args(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    output_path = args.output or output_dir / DEFAULT_OUTPUT_NAME
    report_path = args.report or output_dir / DEFAULT_REPORT_NAME
    smiles_output_path = args.smiles_output or default_smiles_output_path(output_path)
    return output_dir, output_path, smiles_output_path, report_path


def confirm_output_dir_overwrite(output_dir: Path, *, assume_yes: bool) -> None:
    if not output_dir.exists():
        return
    script_dir = Path(__file__).resolve().parent
    if output_dir.resolve() == script_dir:
        raise SystemExit(f"Refusing to delete source directory {output_dir}.")
    if assume_yes:
        shutil.rmtree(output_dir)
        return
    if not sys.stdin.isatty():
        raise SystemExit(f"Output directory already exists: {output_dir}. Run with --yes to overwrite.")
    answer = input(f"Output directory already exists: {output_dir}\nDelete it and overwrite? [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        raise SystemExit("Aborted without deleting existing output directory.")
    shutil.rmtree(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample PubChem molecules for descriptor-bin coverage.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--smiles-output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--descriptor-names", default=",".join(DEFAULT_DESCRIPTOR_NAMES))
    parser.add_argument("--min-count", type=int, default=100)
    parser.add_argument("--max-count", type=int, default=300)
    parser.add_argument("--discovery-rows", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-threads", type=int, default=1, help="Torch/BLAS threads per worker subprocess.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--center", choices=("mean", "median"), default="mean")
    parser.add_argument("--cid-column", default="cid")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--worker-input", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    descriptor_names = parse_csv_strings(args.descriptor_names)
    missing = [name for name in descriptor_names if name not in DEFAULT_DESCRIPTOR_BIN_SPECS_BY_NAME]
    if missing:
        raise SystemExit(f"Missing descriptor bin specs for: {', '.join(missing)}")
    if args.worker_input is not None:
        if args.worker_output is None:
            raise SystemExit("Worker output path is required.")
        run_worker(
            input_path=args.worker_input,
            output_path=args.worker_output,
            descriptor_names=descriptor_names,
            worker_threads=args.worker_threads,
        )
        return

    output_dir, output_path, smiles_output_path, report_path = output_paths_from_args(args)
    confirm_output_dir_overwrite(output_dir, assume_yes=args.yes)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = sample_pubchem_descriptor_coverage(
        input_path=args.input,
        output_path=output_path,
        smiles_output_path=smiles_output_path,
        report_path=report_path,
        descriptor_names=descriptor_names,
        min_count=args.min_count,
        max_count=args.max_count,
        discovery_rows=args.discovery_rows,
        batch_size=args.batch_size,
        workers=args.workers,
        worker_threads=args.worker_threads,
        seed=args.seed,
        center=args.center,
        cid_column=args.cid_column,
        smiles_column=args.smiles_column,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
