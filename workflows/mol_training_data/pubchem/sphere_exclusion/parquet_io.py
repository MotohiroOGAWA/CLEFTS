from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def total_rows(input_path: Path) -> int:
    return pq.ParquetFile(input_path).metadata.num_rows


def iter_randomized_batches(
    input_path: Path,
    *,
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
    for row_group in tqdm(row_groups, desc='row groups', unit='group'):
        batches = parquet_file.iter_batches(
            batch_size=batch_size,
            row_groups=[row_group],
            columns=[cid_column, smiles_column],
        )
        for batch in batches:
            frame = batch.to_pandas()
            if len(frame) == 0:
                continue
            frame = frame.sample(
                frac=1.0,
                random_state=rng.randrange(0, 2**32 - 1),
            ).reset_index(drop=True)
            if max_rows_to_scan is not None:
                remaining = max_rows_to_scan - scanned
                if remaining <= 0:
                    return
                frame = frame.iloc[:remaining]
            scanned += len(frame)
            yield frame


def output_schema(*, include_fingerprint_hex: bool) -> pa.Schema:
    fields = [
        pa.field('CID', pa.int64()),
        pa.field('SMILES', pa.string()),
    ]
    if include_fingerprint_hex:
        fields.append(pa.field('morgan_fp_hex', pa.string()))
    return pa.schema(fields)


def rows_to_table(rows: list[dict[str, object]], *, include_fingerprint_hex: bool) -> pa.Table:
    names = ['CID', 'SMILES']
    if include_fingerprint_hex:
        names.append('morgan_fp_hex')
    arrays = []
    for name in names:
        arrays.append(pa.array([row.get(name) for row in rows]))
    return pa.Table.from_arrays(arrays, names=names)
