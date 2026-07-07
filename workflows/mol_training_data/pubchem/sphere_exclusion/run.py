from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import RDLogger
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fingerprints import fingerprint_to_hex, morgan_fingerprint, mol_from_smiles
    from parquet_io import iter_randomized_batches, output_schema, rows_to_table, total_rows
    from selector import SphereExclusionIndex
else:
    from .fingerprints import fingerprint_to_hex, morgan_fingerprint, mol_from_smiles
    from .parquet_io import iter_randomized_batches, output_schema, rows_to_table, total_rows
    from .selector import SphereExclusionIndex


DEFAULT_INPUT = Path("/workspaces/CLEFTS/raw_data/PubChem/pubchem_cid_smiles.parquet")
DEFAULT_OUTPUT_DIR = Path("/workspaces/CLEFTS/mnt/app/data/processed/pubchem/sphere_exclusion")
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "pubchem_morgan2048_sphere_exclusion.parquet"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "pubchem_morgan2048_sphere_exclusion_report.json"
PROGRESS_POLL_SECONDS = 0.2
WORKER_PROGRESS_INTERVAL = 100


def default_smiles_output_path(output_path: Path) -> Path:
    return output_path.with_suffix(".smiles.txt")


def write_selected_rows(
    writer: pq.ParquetWriter,
    smiles_file,
    rows: list[dict[str, object]],
    *,
    include_fingerprint_hex: bool,
) -> int:
    if not rows:
        return 0
    table = rows_to_table(rows, include_fingerprint_hex=include_fingerprint_hex)
    writer.write_table(table)
    for row in rows:
        smiles = str(row.get("SMILES", "")).strip()
        if smiles:
            smiles_file.write(f"{smiles}\n")
    written = len(rows)
    rows.clear()
    return written


def fingerprint_record(
    record: tuple[object, object],
    *,
    radius: int,
    n_bits: int,
) -> tuple[object, object, object]:
    cid, smiles = record
    mol = mol_from_smiles(smiles)
    return cid, smiles, morgan_fingerprint(mol, radius=radius, n_bits=n_bits)


def write_worker_input(
    path: Path,
    indexed_records: list[tuple[int, object, object]],
) -> None:
    table = pa.Table.from_arrays(
        [
            pa.array([int(result_index) for result_index, _, _ in indexed_records], type=pa.int64()),
            pa.array([int(cid) for _, cid, _ in indexed_records], type=pa.int64()),
            pa.array([str(smiles) for _, _, smiles in indexed_records], type=pa.string()),
        ],
        names=["result_index", "cid", "smiles"],
    )
    pq.write_table(table, path, compression="zstd")


def run_fingerprint_worker(
    *,
    input_path: Path,
    output_path: Path,
    progress_path: Path,
    radius: int,
    n_bits: int,
    threshold: float | None,
    leader_snapshot_path: Path | None,
) -> None:
    RDLogger.DisableLog("rdApp.*")
    table = pq.read_table(input_path, columns=["result_index", "cid", "smiles"])
    result_indices = table.column("result_index").to_pylist()
    cids = table.column("cid").to_pylist()
    smiles_values = table.column("smiles").to_pylist()
    results = []
    prefiltered_excluded = 0
    max_worker_similarity = 0.0
    worker_comparisons = 0
    worker_index = None
    if leader_snapshot_path is not None and threshold is not None:
        worker_index = SphereExclusionIndex(threshold=threshold)
        with open(leader_snapshot_path, "rb") as f:
            for leader_fp in pickle.load(f):
                worker_index.add(leader_fp)

    progress_path.write_text("0", encoding="utf-8")

    for count, (result_index, cid, smiles) in enumerate(
        zip(result_indices, cids, smiles_values),
        start=1,
    ):
        _, _, fp = fingerprint_record((cid, smiles), radius=radius, n_bits=n_bits)
        if fp is not None and worker_index is not None:
            before_comparisons = worker_index.comparison_count
            excluded, similarity = worker_index.is_excluded(fp)
            worker_comparisons += worker_index.comparison_count - before_comparisons
            max_worker_similarity = max(max_worker_similarity, similarity)
            if excluded:
                prefiltered_excluded += 1
                if count % WORKER_PROGRESS_INTERVAL == 0:
                    progress_path.write_text(str(count), encoding="utf-8")
                continue
        results.append((int(result_index), cid, smiles, fp))
        if count % WORKER_PROGRESS_INTERVAL == 0:
            progress_path.write_text(str(count), encoding="utf-8")

    progress_path.write_text(str(len(result_indices)), encoding="utf-8")
    payload = {
        "results": results,
        "processed": len(result_indices),
        "prefiltered_excluded": prefiltered_excluded,
        "max_worker_similarity": max_worker_similarity,
        "worker_comparisons": worker_comparisons,
    }
    with open(output_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_worker_progress(progress_paths: list[Path]) -> int:
    total = 0
    for path in progress_paths:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue
        if text:
            total += int(text)
    return total


def chunk_indexed_records(
    records: list[tuple[object, object]],
    *,
    worker_count: int,
) -> list[list[tuple[int, object, object]]]:
    if not records:
        return []
    worker_count = max(1, min(worker_count, len(records)))
    chunk_size = math.ceil(len(records) / worker_count)
    indexed_records = [(index, cid, smiles) for index, (cid, smiles) in enumerate(records)]
    return [
        indexed_records[start : start + chunk_size]
        for start in range(0, len(indexed_records), chunk_size)
    ]


def run_fingerprint_subprocesses(
    *,
    records: list[tuple[object, object]],
    window_index: int,
    worker_count: int,
    radius: int,
    n_bits: int,
    threshold: float,
    leader_fingerprints: list[object],
    work_dir: Path,
) -> tuple[list[tuple[object, object, object]], dict[str, int | float]]:
    worker_count = max(worker_count, 1)
    window_dir = work_dir / f"window_{window_index:06d}"
    window_dir.mkdir(parents=True, exist_ok=True)

    leader_snapshot_path = None
    if leader_fingerprints:
        leader_snapshot_path = window_dir / "leader_snapshot.pkl"
        with open(leader_snapshot_path, "wb") as f:
            pickle.dump(leader_fingerprints, f, protocol=pickle.HIGHEST_PROTOCOL)

    chunks = chunk_indexed_records(records, worker_count=worker_count)
    processes: list[tuple[subprocess.Popen[bytes], Path, Path, Path, object]] = []
    progress_paths: list[Path] = []
    script_path = Path(__file__).resolve()

    for worker_index, chunk in enumerate(chunks):
        worker_input = window_dir / f"worker_{worker_index:03d}.parquet"
        worker_output = window_dir / f"worker_{worker_index:03d}.pkl"
        worker_progress = window_dir / f"worker_{worker_index:03d}.progress"
        worker_stderr = window_dir / f"worker_{worker_index:03d}.stderr.log"
        stderr_file = open(worker_stderr, "wb")
        write_worker_input(worker_input, chunk)
        progress_paths.append(worker_progress)
        process = subprocess.Popen(
            [
                sys.executable,
                str(script_path),
                "--fingerprint-worker-input",
                str(worker_input),
                "--fingerprint-worker-output",
                str(worker_output),
                "--fingerprint-worker-progress",
                str(worker_progress),
                "--radius",
                str(radius),
                "--n-bits",
                str(n_bits),
                "--threshold",
                str(threshold),
            ]
            + (
                ["--fingerprint-worker-leaders", str(leader_snapshot_path)]
                if leader_snapshot_path is not None
                else []
            ),
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
        )
        processes.append((process, worker_output, worker_progress, worker_stderr, stderr_file))

    with tqdm(
        total=len(records),
        desc=f"window {window_index:06d} Morgan FP",
        unit="mol",
        leave=False,
        mininterval=1.0,
    ) as fingerprint_progress:
        last_progress = 0
        pending = set(range(len(processes)))
        while pending:
            current_progress = read_worker_progress(progress_paths)
            if current_progress > last_progress:
                fingerprint_progress.update(current_progress - last_progress)
                last_progress = current_progress

            finished = []
            for process_index in pending:
                process, _, _, worker_stderr, stderr_file = processes[process_index]
                return_code = process.poll()
                if return_code is None:
                    continue
                if return_code != 0:
                    process.wait()
                    stderr_file.close()
                    message = worker_stderr.read_text(encoding="utf-8", errors="replace")
                    raise RuntimeError(
                        f"Fingerprint worker {process_index} failed with exit code "
                        f"{return_code}: {message}"
                    )
                stderr_file.close()
                finished.append(process_index)
            pending.difference_update(finished)
            if pending:
                time.sleep(PROGRESS_POLL_SECONDS)

        current_progress = read_worker_progress(progress_paths)
        if current_progress < len(records):
            current_progress = len(records)
        if current_progress > last_progress:
            fingerprint_progress.update(current_progress - last_progress)

    ordered_results: list[tuple[object, object, object] | None] = [None] * len(records)
    prefiltered_excluded = 0
    max_worker_similarity = 0.0
    worker_comparisons = 0
    for _, worker_output, _, _, _ in processes:
        with open(worker_output, "rb") as f:
            payload = pickle.load(f)
        prefiltered_excluded += int(payload.get("prefiltered_excluded", 0))
        max_worker_similarity = max(
            max_worker_similarity,
            float(payload.get("max_worker_similarity", 0.0)),
        )
        worker_comparisons += int(payload.get("worker_comparisons", 0))
        for result_index, cid, smiles, fp in payload["results"]:
            ordered_results[int(result_index)] = (cid, smiles, fp)

    filtered_results = [
        result for result in ordered_results if result is not None
    ]
    stats = {
        "prefiltered_excluded": prefiltered_excluded,
        "max_worker_similarity": max_worker_similarity,
        "worker_comparisons": worker_comparisons,
    }
    shutil.rmtree(window_dir, ignore_errors=True)
    return filtered_results, stats


def confirm_output_dir_overwrite(output_dir: Path, *, assume_yes: bool) -> None:
    if not output_dir.exists():
        return

    script_dir = Path(__file__).resolve().parent
    if output_dir.resolve() == script_dir:
        raise SystemExit(
            f"Refusing to delete source directory {output_dir}. "
            "Please set --output to a dedicated output directory."
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


def run_sphere_exclusion(
    *,
    input_path: Path,
    output_path: Path,
    smiles_output_path: Path | None,
    report_path: Path | None,
    threshold: float,
    radius: int,
    n_bits: int,
    batch_size: int,
    write_batch_size: int,
    seed: int,
    max_rows_to_scan: int | None,
    max_selected: int | None,
    include_fingerprint_hex: bool,
    cid_column: str,
    smiles_column: str,
    progress_update_interval: int,
    fingerprint_workers: int | None = None,
) -> dict[str, Any]:
    RDLogger.DisableLog("rdApp.*")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    smiles_output_path = smiles_output_path or default_smiles_output_path(output_path)
    smiles_output_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    index = SphereExclusionIndex(threshold=threshold)
    writer = pq.ParquetWriter(
        output_path,
        output_schema(include_fingerprint_hex=include_fingerprint_hex),
        compression="zstd",
    )
    smiles_file = open(smiles_output_path, "w", encoding="utf-8")

    input_rows = total_rows(input_path)
    scan_total = min(input_rows, max_rows_to_scan) if max_rows_to_scan is not None else input_rows
    progress_update_interval = max(progress_update_interval, 1)
    if fingerprint_workers is None:
        fingerprint_workers = os.cpu_count() or 1
    fingerprint_workers = max(fingerprint_workers, 1)
    fingerprint_window_size = batch_size * fingerprint_workers
    fingerprint_temp_dir = tempfile.TemporaryDirectory(
        prefix="fingerprint_workers_",
        dir=str(output_path.parent),
    )
    fingerprint_work_dir = Path(fingerprint_temp_dir.name)

    scanned_rows = 0
    valid_fingerprints = 0
    invalid_smiles = 0
    selected_rows = 0
    excluded_rows = 0
    prefiltered_excluded_rows = 0
    worker_tanimoto_comparisons = 0
    max_observed_similarity = 0.0
    buffer: list[dict[str, object]] = []

    scan_progress = tqdm(total=scan_total, desc="sphere exclusion", unit="record", position=0)
    selected_progress = None
    if max_selected is not None:
        selected_progress = tqdm(
            total=max_selected,
            desc="selected records",
            unit="record",
            position=1,
        )

    def update_postfix() -> None:
        scan_progress.set_postfix(
            excluded=excluded_rows,
            invalid=invalid_smiles,
            leaders=index.selected_count,
            buckets=index.bucket_count,
            refresh=False,
        )
        if selected_progress is not None:
            selected_progress.set_postfix(
                scanned=scanned_rows,
                excluded=excluded_rows,
                refresh=False,
            )

    def process_fingerprint_window(
        records: list[tuple[object, object]],
        *,
        window_index: int,
    ) -> bool:
        nonlocal scanned_rows
        nonlocal valid_fingerprints
        nonlocal invalid_smiles
        nonlocal selected_rows
        nonlocal excluded_rows
        nonlocal prefiltered_excluded_rows
        nonlocal worker_tanimoto_comparisons
        nonlocal max_observed_similarity

        fingerprint_results, worker_stats = run_fingerprint_subprocesses(
            records=records,
            window_index=window_index,
            worker_count=fingerprint_workers,
            radius=radius,
            n_bits=n_bits,
            threshold=threshold,
            leader_fingerprints=index.fingerprints(),
            work_dir=fingerprint_work_dir,
        )
        worker_excluded = int(worker_stats["prefiltered_excluded"])
        prefiltered_excluded_rows += worker_excluded
        excluded_rows += worker_excluded
        scanned_rows += worker_excluded
        scan_progress.update(worker_excluded)
        worker_tanimoto_comparisons += int(worker_stats["worker_comparisons"])
        max_observed_similarity = max(
            max_observed_similarity,
            float(worker_stats["max_worker_similarity"]),
        )

        for cid, smiles, fp in tqdm(
            fingerprint_results,
            total=len(fingerprint_results),
            desc=f"window {window_index:06d} sphere exclusion",
            unit="mol",
            leave=False,
            mininterval=1.0,
        ):
            scanned_rows += 1
            scan_progress.update(1)

            if fp is None:
                invalid_smiles += 1
                if scanned_rows % progress_update_interval == 0:
                    update_postfix()
                continue

            valid_fingerprints += 1
            excluded, similarity = index.is_excluded(fp)
            max_observed_similarity = max(max_observed_similarity, similarity)
            if excluded:
                excluded_rows += 1
            else:
                index.add(fp)
                selected_row: dict[str, object] = {
                    "CID": int(cid),
                    "SMILES": str(smiles),
                }
                if include_fingerprint_hex:
                    selected_row["morgan_fp_hex"] = fingerprint_to_hex(fp)
                buffer.append(selected_row)
                selected_rows += 1
                if selected_progress is not None:
                    selected_progress.update(1)
                if len(buffer) >= write_batch_size:
                    write_selected_rows(
                        writer,
                        smiles_file,
                        buffer,
                        include_fingerprint_hex=include_fingerprint_hex,
                    )

            if scanned_rows % progress_update_interval == 0:
                update_postfix()

            if max_selected is not None and selected_rows >= max_selected:
                return True

        return False

    update_postfix()
    try:
        pending_records: list[tuple[object, object]] = []
        window_index = 0
        for frame in iter_randomized_batches(
            input_path,
            cid_column=cid_column,
            smiles_column=smiles_column,
            batch_size=batch_size,
            rng=rng,
            max_rows_to_scan=max_rows_to_scan,
        ):
            cids = frame[cid_column].to_numpy()
            smiles_values = frame[smiles_column].to_numpy()
            pending_records.extend(zip(cids, smiles_values))

            if len(pending_records) < fingerprint_window_size:
                continue

            should_stop = process_fingerprint_window(
                pending_records,
                window_index=window_index,
            )
            pending_records = []
            window_index += 1
            update_postfix()
            if should_stop:
                break
        else:
            if pending_records:
                process_fingerprint_window(
                    pending_records,
                    window_index=window_index,
                )
                update_postfix()
    finally:
        write_selected_rows(writer, smiles_file, buffer, include_fingerprint_hex=include_fingerprint_hex)
        smiles_file.close()
        writer.close()
        fingerprint_temp_dir.cleanup()
        update_postfix()
        scan_progress.close()
        if selected_progress is not None:
            selected_progress.close()

    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "smiles_output_path": str(smiles_output_path),
        "threshold": threshold,
        "radius": radius,
        "n_bits": n_bits,
        "batch_size": batch_size,
        "fingerprint_window_size": fingerprint_window_size,
        "write_batch_size": write_batch_size,
        "seed": seed,
        "max_rows_to_scan": max_rows_to_scan,
        "max_selected": max_selected,
        "include_fingerprint_hex": include_fingerprint_hex,
        "fingerprint_workers": fingerprint_workers,
        "fingerprint_parallelism": "subprocess",
        "scanned_rows": scanned_rows,
        "valid_fingerprints": valid_fingerprints,
        "invalid_smiles": invalid_smiles,
        "selected_rows": selected_rows,
        "excluded_rows": excluded_rows,
        "prefiltered_excluded_rows": prefiltered_excluded_rows,
        "leader_count": index.selected_count,
        "bucket_count": index.bucket_count,
        "tanimoto_comparisons": index.comparison_count,
        "worker_tanimoto_comparisons": worker_tanimoto_comparisons,
        "max_observed_similarity_to_leader": max_observed_similarity,
    }

    if report_path is not None:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Morgan 2048-bit sphere-exclusion subset from PubChem CID/SMILES parquet."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smiles-output", type=Path, default=None, help="Newline-delimited SMILES output. Defaults to the output path with .smiles.txt suffix.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--threshold", type=float, default=0.65, help="Tanimoto exclusion threshold, commonly 0.6-0.7.")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--write-batch-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows-to-scan", type=int, default=None)
    parser.add_argument("--max-selected", type=int, default=None, help="Stop after this many selected molecules. Useful for multi-million training sets.")
    parser.add_argument("--include-fingerprint-hex", action="store_true")
    parser.add_argument("--cid-column", default="cid")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--progress-update-interval", type=int, default=1000)
    parser.add_argument(
        "--fingerprint-workers",
        type=int,
        default=1,
        help="Number of subprocesses for SMILES parsing and Morgan fingerprint generation. Defaults to 1.",
    )
    parser.add_argument("--yes", action="store_true", help="Delete an existing output directory without prompting.")
    parser.add_argument("--fingerprint-worker-input", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fingerprint-worker-output", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fingerprint-worker-progress", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fingerprint-worker-leaders", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fingerprint_worker_input is not None:
        if args.fingerprint_worker_output is None or args.fingerprint_worker_progress is None:
            raise SystemExit("Fingerprint worker output and progress paths are required.")
        run_fingerprint_worker(
            input_path=args.fingerprint_worker_input,
            output_path=args.fingerprint_worker_output,
            progress_path=args.fingerprint_worker_progress,
            radius=args.radius,
            n_bits=args.n_bits,
            threshold=args.threshold,
            leader_snapshot_path=args.fingerprint_worker_leaders,
        )
        return

    confirm_output_dir_overwrite(args.output.parent, assume_yes=args.yes)
    report = run_sphere_exclusion(
        input_path=args.input,
        output_path=args.output,
        smiles_output_path=args.smiles_output,
        report_path=args.report,
        threshold=args.threshold,
        radius=args.radius,
        n_bits=args.n_bits,
        batch_size=args.batch_size,
        write_batch_size=args.write_batch_size,
        seed=args.seed,
        max_rows_to_scan=args.max_rows_to_scan,
        max_selected=args.max_selected,
        include_fingerprint_hex=args.include_fingerprint_hex,
        cid_column=args.cid_column,
        smiles_column=args.smiles_column,
        progress_update_interval=args.progress_update_interval,
        fingerprint_workers=args.fingerprint_workers,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
