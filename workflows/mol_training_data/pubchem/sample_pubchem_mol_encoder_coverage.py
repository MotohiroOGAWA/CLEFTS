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
from typing import Dict, Iterable, List, Sequence, Set, Tuple

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
from clefts.ml.mol.atom_feature import AtomFeatureLayer  # noqa: E402
from clefts.ml.mol.bond_feature import BondFeatureLayer  # noqa: E402
from clefts.ml.mol.graph_builder import MolGraphBuilder  # noqa: E402
from clefts.utils.parallel_subprocess import run_parallel_subprocesses  # noqa: E402


DEFAULT_INPUT = Path("/workspaces/CLEFTS/raw_data/PubChem/pubchem_cid_smiles.parquet")
DEFAULT_OUTPUT_DIR = Path("/workspaces/CLEFTS/mnt/app/data/processed/pubchem/mol_encoder_coverage_sample")
DEFAULT_OUTPUT_NAME = "pubchem_mol_encoder_coverage_sample.parquet"
DEFAULT_REPORT_NAME = "pubchem_mol_encoder_coverage_sample_report.json"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_NAME
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / DEFAULT_REPORT_NAME

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


def default_smiles_output_path(output_path: Path) -> Path:
    return output_path.with_suffix(".smiles.txt")


def smiles_lines(values: Sequence[object]) -> List[str]:
    lines: List[str] = []
    for value in values:
        if pd.isna(value):
            continue
        smiles = str(value).strip()
        if smiles:
            lines.append(smiles)
    return lines


def write_smiles_file(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for smiles in smiles_lines(frame[SMILES_OUTPUT]):
            f.write(f"{smiles}\n")


def safe_label(value: object) -> str:
    label = str(value).strip()
    if label.startswith("-"):
        label = f"minus_{label[1:]}"
    elif label.startswith("+"):
        label = f"plus_{label[1:]}"
    replacements = (
        (" ", "_"),
        ("/", "_"),
        ("\\", "_"),
        (",", "_"),
        (":", "_"),
        ("-", "_"),
        ("+", "_"),
    )
    for old, new in replacements:
        label = label.replace(old, new)
    return label


def column_name(kind: str, group: str, value: object) -> str:
    return f"{kind}_{group}_{safe_label(value)}"


def feature_columns_from_sets(kind: str, feature_sets: Dict[str, Tuple[object, ...]]) -> List[str]:
    columns: List[str] = []
    for group, values in feature_sets.items():
        columns.extend(column_name(kind, group, value) for value in values)
    return columns


def atom_feature_columns(symbols: Sequence[str]) -> List[str]:
    layer = AtomFeatureLayer(symbols=tuple(symbols))
    return feature_columns_from_sets("atom", layer.feature_sets)


def bond_feature_columns() -> List[str]:
    return feature_columns_from_sets("bond", BondFeatureLayer().feature_sets)


def target_feature_columns(symbols: Sequence[str]) -> List[str]:
    return [*atom_feature_columns(symbols), *bond_feature_columns()]


def parse_symbols(value: str | Sequence[str]) -> Tuple[str, ...]:
    if isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = []
        for item in value:
            raw_items.extend(str(item).split(","))
    symbols = tuple(sorted({item.strip() for item in raw_items if item.strip()}))
    if not symbols:
        raise ValueError("At least one symbol must be specified.")
    return symbols


def max_feature_flags(feature_tensor: torch.Tensor, feature_dim: int) -> List[bool]:
    if feature_tensor.numel() == 0:
        return [False] * feature_dim
    return (feature_tensor.max(dim=0).values > 0).cpu().tolist()


def compounds_from_smiles(
    smiles_values: Sequence[object],
    *,
    desc: str,
    show_progress: bool = True,
) -> Tuple[List[Compound | None], int, Tuple[str, ...]]:
    compounds: List[Compound | None] = []
    symbols: Set[str] = set()
    invalid_smiles = 0

    for smiles in tqdm(
        smiles_values,
        desc=desc,
        unit="smiles",
        leave=False,
        mininterval=1.0,
        disable=not show_progress,
    ):
        if pd.isna(smiles):
            compounds.append(None)
            invalid_smiles += 1
            continue
        try:
            compound = Compound.from_smiles(str(smiles))
        except Exception:
            compounds.append(None)
            invalid_smiles += 1
            continue
        compounds.append(compound)
        symbols.update(atom.GetSymbol() for atom in compound.mol.GetAtoms())

    return compounds, invalid_smiles, tuple(sorted(symbols))


def compound_feature_flags(
    compound: Compound | None,
    builder: MolGraphBuilder,
    builder_columns: Sequence[str],
    target_columns: Sequence[str],
) -> Dict[str, bool]:
    if compound is None:
        return {column: False for column in target_columns}

    data = builder.build(compound)
    flags = [
        *max_feature_flags(data.x, builder.atom_dim),
        *max_feature_flags(data.edge_attr, builder.bond_dim),
    ]
    by_column = dict(zip(builder_columns, flags))
    return {column: bool(by_column.get(column, False)) for column in target_columns}


def iter_randomized_batches(
    input_path: Path,
    cid_column: str,
    smiles_column: str,
    batch_size: int,
    rng: random.Random,
    max_rows_to_scan: int | None,
) -> Iterable[pd.DataFrame]:
    parquet_file = pq.ParquetFile(input_path)
    row_groups = list(range(parquet_file.metadata.num_row_groups))
    rng.shuffle(row_groups)

    scanned = 0
    for row_group in tqdm(row_groups, desc="row groups", unit="group"):
        batches = parquet_file.iter_batches(
            batch_size=batch_size,
            row_groups=[row_group],
            columns=[cid_column, smiles_column],
        )
        for batch in batches:
            frame = batch.to_pandas()
            if len(frame) == 0:
                continue
            random_state = rng.randrange(0, 2**32 - 1)
            frame = frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
            if max_rows_to_scan is not None:
                remaining = max_rows_to_scan - scanned
                if remaining <= 0:
                    return
                frame = frame.iloc[:remaining]
            scanned += len(frame)
            yield frame


def write_output(
    rows: Sequence[Dict[str, object]],
    output_path: Path,
    smiles_output_path: Path,
    feature_columns: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=[CID_OUTPUT, SMILES_OUTPUT, *feature_columns])
    for column in feature_columns:
        if column not in frame.columns:
            frame[column] = False
        frame[column] = frame[column].astype(bool)
    frame = frame[[CID_OUTPUT, SMILES_OUTPUT, *feature_columns]]
    frame.to_parquet(output_path, index=False, compression="zstd")
    write_smiles_file(frame, smiles_output_path)


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


def worker_batch_candidates(
    frame: pd.DataFrame,
    *,
    symbols: Sequence[str],
    min_count: int,
    feature_columns: Sequence[str],
    show_progress: bool = True,
) -> Dict[str, object]:
    selected_symbols = tuple(sorted(set(symbols)))
    counts = {column: 0 for column in feature_columns}
    selected_rows: List[Dict[str, object]] = []
    invalid_smiles = 0
    processed_rows = 0

    compounds, batch_invalid_smiles, batch_symbols = compounds_from_smiles(
        frame[SMILES_OUTPUT],
        desc="worker SMILES to Compound",
        show_progress=show_progress,
    )
    invalid_smiles += batch_invalid_smiles

    builder_symbols = tuple(sorted(set(selected_symbols).union(batch_symbols)))
    builder = MolGraphBuilder(symbols=builder_symbols)
    builder_columns = [*atom_feature_columns(builder.symbols), *bond_feature_columns()]

    for row_index, compound in enumerate(
        tqdm(
            compounds,
            desc="worker graph features",
            unit="compound",
            leave=False,
            mininterval=1.0,
            disable=not show_progress,
        )
    ):
        processed_rows += 1
        flags = compound_feature_flags(compound, builder, builder_columns, feature_columns)
        needed_columns = [column for column, value in flags.items() if value and counts[column] < min_count]
        if needed_columns:
            source_row = frame.iloc[row_index]
            selected_row: Dict[str, object] = {
                "result_index": int(source_row["result_index"]),
                CID_OUTPUT: source_row[CID_OUTPUT],
                SMILES_OUTPUT: source_row[SMILES_OUTPUT],
            }
            selected_row.update(flags)
            selected_rows.append(selected_row)
            for column, value in flags.items():
                if value:
                    counts[column] += 1
    return {
        "rows": selected_rows,
        "processed_rows": processed_rows,
        "invalid_smiles": invalid_smiles,
        "feature_counts": counts,
    }


def run_worker(
    *,
    input_path: Path,
    output_path: Path,
    symbols: Sequence[str],
    min_count: int,
    worker_threads: int,
) -> None:
    RDLogger.DisableLog("rdApp.*")
    configure_torch_threads(worker_threads)
    table = pq.read_table(input_path)
    frame = table.to_pandas()
    feature_columns = target_feature_columns(symbols)
    payload = worker_batch_candidates(
        frame,
        symbols=symbols,
        min_count=min_count,
        feature_columns=feature_columns,
        show_progress=False,
    )
    with open(output_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_worker_wave(
    *,
    frames: Sequence[pd.DataFrame],
    wave_index: int,
    worker_count: int,
    symbols: Sequence[str],
    min_count: int,
    work_dir: Path,
    result_offset: int,
    worker_threads: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object], int]:
    if worker_count == 1:
        candidate_rows: List[Dict[str, object]] = []
        processed_rows = 0
        invalid_smiles = 0
        next_result_offset = result_offset
        feature_columns = target_feature_columns(symbols)
        for frame in frames:
            worker_frame = frame.rename(columns={"cid": CID_OUTPUT, "smiles": SMILES_OUTPUT}).copy()
            worker_frame.insert(
                0,
                "result_index",
                range(next_result_offset, next_result_offset + len(worker_frame)),
            )
            next_result_offset += len(worker_frame)
            payload = worker_batch_candidates(
                worker_frame,
                symbols=symbols,
                min_count=min_count,
                feature_columns=feature_columns,
            )
            candidate_rows.extend(payload["rows"])
            processed_rows += int(payload["processed_rows"])
            invalid_smiles += int(payload["invalid_smiles"])
        candidate_rows.sort(key=lambda row: int(row["result_index"]))
        return candidate_rows, {"processed_rows": processed_rows, "invalid_smiles": invalid_smiles}, next_result_offset

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
                    "--symbols",
                    ",".join(symbols),
                    "--min-count",
                    str(min_count),
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
            desc=f"wave {wave_index:06d} worker coverage",
        )

        candidate_rows: List[Dict[str, object]] = []
        processed_rows = 0
        invalid_smiles = 0
        for worker_output in worker_outputs:
            with open(worker_output, "rb") as f:
                payload = pickle.load(f)
            candidate_rows.extend(payload["rows"])
            processed_rows += int(payload["processed_rows"])
            invalid_smiles += int(payload["invalid_smiles"])
        candidate_rows.sort(key=lambda row: int(row["result_index"]))
        return candidate_rows, {"processed_rows": processed_rows, "invalid_smiles": invalid_smiles}, next_result_offset
    finally:
        shutil.rmtree(wave_dir, ignore_errors=True)


def candidate_needed_columns(
    row: Dict[str, object],
    counts: Dict[str, int],
    *,
    min_count: int,
    max_count: int | None,
    feature_columns: Sequence[str],
) -> List[str]:
    columns = []
    for column in feature_columns:
        if not bool(row.get(column, False)):
            continue
        if counts[column] >= min_count:
            continue
        if max_count is not None and counts[column] >= max_count:
            continue
        columns.append(column)
    return columns


def output_paths_from_args(args: argparse.Namespace) -> tuple[Path, Path, Path | None, Path | None]:
    output_dir = args.output_dir
    output_path = args.output
    report_path = args.report
    smiles_output_path = args.smiles_output

    if output_dir is not None:
        output_path = output_path or output_dir / DEFAULT_OUTPUT_NAME
        report_path = report_path or output_dir / DEFAULT_REPORT_NAME
        smiles_output_path = smiles_output_path or default_smiles_output_path(output_path)
    else:
        output_path = output_path or DEFAULT_OUTPUT
        report_path = report_path or DEFAULT_REPORT
        smiles_output_path = smiles_output_path or default_smiles_output_path(output_path)
        output_dir = output_path.parent

    return output_path, smiles_output_path, report_path, output_dir


def confirm_output_dir_overwrite(output_dir: Path, *, assume_yes: bool) -> None:
    if not output_dir.exists():
        return

    script_dir = Path(__file__).resolve().parent
    if output_dir.resolve() == script_dir:
        raise SystemExit(
            f"Refusing to delete source directory {output_dir}. "
            "Please set --output-dir to a dedicated output directory."
        )

    if assume_yes:
        shutil.rmtree(output_dir)
        return

    if not sys.stdin.isatty():
        raise SystemExit(
            f"Output directory already exists: {output_dir}\n"
            "Run again with --yes to delete it and overwrite non-interactively."
        )

    answer = input(
        f"Output directory already exists: {output_dir}\n"
        "Delete it and overwrite? [y/N] "
    ).strip().lower()
    if answer not in {"y", "yes"}:
        raise SystemExit("Aborted without deleting existing output directory.")
    shutil.rmtree(output_dir)


def sample_pubchem_coverage(
    input_path: Path,
    output_path: Path,
    smiles_output_path: Path | None,
    report_path: Path | None,
    symbols: Sequence[str],
    min_count: int,
    max_count: int | None,
    batch_size: int,
    workers: int,
    worker_threads: int,
    seed: int,
    max_rows_to_scan: int | None,
    progress_update_interval: int,
    cid_column: str,
    smiles_column: str,
) -> Dict[str, object]:
    RDLogger.DisableLog("rdApp.*")
    if max_count is not None and max_count < min_count:
        raise ValueError("max_count must be greater than or equal to min_count.")

    smiles_output_path = smiles_output_path or default_smiles_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    smiles_output_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    selected_symbols = tuple(sorted(set(symbols)))
    feature_columns = target_feature_columns(selected_symbols)
    counts = {column: 0 for column in feature_columns}
    selected_rows: List[Dict[str, object]] = []
    invalid_smiles = 0
    scanned_rows = 0
    selected_count = 0
    workers = max(workers, 1)
    worker_threads = max(worker_threads, 1)
    work_temp_dir = tempfile.TemporaryDirectory(prefix="coverage_workers_", dir=str(output_path.parent))
    work_dir = Path(work_temp_dir.name)

    def complete() -> bool:
        return all(count >= min_count for count in counts.values())

    def incomplete_feature_count() -> int:
        return sum(count < min_count for count in counts.values())

    def remaining_needed_sum() -> int:
        return sum(max(min_count - count, 0) for count in counts.values())

    input_rows = pq.ParquetFile(input_path).metadata.num_rows
    scan_total = min(input_rows, max_rows_to_scan) if max_rows_to_scan is not None else input_rows
    progress_update_interval = max(progress_update_interval, 1)
    scan_progress = tqdm(total=scan_total, desc="scan shuffled records", unit="record")

    def update_scan_postfix() -> None:
        scan_progress.set_postfix(
            records=selected_count,
            scanned=scanned_rows,
            remaining=incomplete_feature_count(),
            remaining_sum=remaining_needed_sum(),
            refresh=False,
        )

    def accept_candidate(row: Dict[str, object]) -> bool:
        nonlocal selected_count
        needed_columns = candidate_needed_columns(
            row,
            counts,
            min_count=min_count,
            max_count=max_count,
            feature_columns=feature_columns,
        )
        if not needed_columns:
            return False
        selected_row: Dict[str, object] = {
            CID_OUTPUT: row[CID_OUTPUT],
            SMILES_OUTPUT: row[SMILES_OUTPUT],
        }
        for column in feature_columns:
            selected_row[column] = bool(row.get(column, False))
        selected_rows.append(selected_row)
        selected_count += 1
        for column, value in selected_row.items():
            if column in counts and bool(value):
                counts[column] += 1
        return True

    update_scan_postfix()
    try:
        pending_frames: List[pd.DataFrame] = []
        wave_index = 0
        result_offset = 0
        for frame in iter_randomized_batches(
            input_path=input_path,
            cid_column=cid_column,
            smiles_column=smiles_column,
            batch_size=batch_size,
            rng=rng,
            max_rows_to_scan=max_rows_to_scan,
        ):
            pending_frames.append(frame.rename(columns={cid_column: CID_OUTPUT, smiles_column: SMILES_OUTPUT}))
            if len(pending_frames) < workers:
                continue

            candidates, stats, result_offset = run_worker_wave(
                frames=pending_frames,
                wave_index=wave_index,
                worker_count=workers,
                symbols=selected_symbols,
                min_count=min_count,
                work_dir=work_dir,
                result_offset=result_offset,
                worker_threads=worker_threads,
            )
            pending_frames = []
            wave_index += 1
            scanned_rows += int(stats["processed_rows"])
            invalid_smiles += int(stats["invalid_smiles"])
            scan_progress.update(int(stats["processed_rows"]))
            for row in candidates:
                accept_candidate(row)
                if complete():
                    break
            update_scan_postfix()
            if complete():
                break
            if scanned_rows % progress_update_interval == 0:
                update_scan_postfix()
        else:
            if pending_frames:
                candidates, stats, result_offset = run_worker_wave(
                    frames=pending_frames,
                    wave_index=wave_index,
                    worker_count=workers,
                    symbols=selected_symbols,
                    min_count=min_count,
                    work_dir=work_dir,
                    result_offset=result_offset,
                    worker_threads=worker_threads,
                )
                scanned_rows += int(stats["processed_rows"])
                invalid_smiles += int(stats["invalid_smiles"])
                scan_progress.update(int(stats["processed_rows"]))
                for row in candidates:
                    accept_candidate(row)
                    if complete():
                        break
                update_scan_postfix()
    finally:
        scan_progress.close()
        work_temp_dir.cleanup()

    write_output(selected_rows, output_path, smiles_output_path, feature_columns)

    incomplete_counts = {column: count for column, count in counts.items() if count < min_count}
    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "smiles_output_path": str(smiles_output_path),
        "symbols": list(selected_symbols),
        "min_count": min_count,
        "max_count": max_count,
        "seed": seed,
        "batch_size": batch_size,
        "workers": workers,
        "worker_threads": worker_threads,
        "max_rows_to_scan": max_rows_to_scan,
        "progress_update_interval": progress_update_interval,
        "scanned_rows": scanned_rows,
        "selected_rows": selected_count,
        "invalid_smiles": invalid_smiles,
        "complete": not incomplete_counts,
        "feature_counts": counts,
        "incomplete_feature_counts": incomplete_counts,
        "feature_columns": feature_columns,
    }

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample PubChem CID/SMILES until each mol-encoder feature has at least N selected examples."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--smiles-output", type=Path, default=None, help="Newline-delimited SMILES output. Defaults to the output path with .smiles.txt suffix.")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--symbols", nargs="+", default=["C,N,O,P,S,F,Cl,Br,I"])
    parser.add_argument("--min-count", type=int, default=100)
    parser.add_argument("--max-count", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-threads", type=int, default=1, help="Torch/BLAS threads per worker subprocess.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows-to-scan", type=int, default=None)
    parser.add_argument("--progress-update-interval", type=int, default=1000)
    parser.add_argument("--cid-column", default="cid")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--yes", action="store_true", help="Delete an existing output directory without prompting.")
    parser.add_argument("--worker-input", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = parse_symbols(args.symbols)
    if args.worker_input is not None:
        if args.worker_output is None:
            raise SystemExit("Worker output path is required.")
        run_worker(
            input_path=args.worker_input,
            output_path=args.worker_output,
            symbols=symbols,
            min_count=args.min_count,
            worker_threads=args.worker_threads,
        )
        return

    output_path, smiles_output_path, report_path, output_dir = output_paths_from_args(args)
    confirm_output_dir_overwrite(output_dir, assume_yes=args.yes)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = sample_pubchem_coverage(
        input_path=args.input,
        output_path=output_path,
        smiles_output_path=smiles_output_path,
        report_path=report_path,
        symbols=symbols,
        min_count=args.min_count,
        max_count=args.max_count,
        batch_size=args.batch_size,
        workers=args.workers,
        worker_threads=args.worker_threads,
        seed=args.seed,
        max_rows_to_scan=args.max_rows_to_scan,
        progress_update_interval=args.progress_update_interval,
        cid_column=args.cid_column,
        smiles_column=args.smiles_column,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
