from __future__ import annotations

import io
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


SUPPORTED_SUFFIXES = (".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".parquet", ".mssim")


def _plain_table(path: Path) -> pd.DataFrame:
    """Read a table that is not an msentity container."""
    suffix = path.suffix.lower()
    if suffix == ".msds":
        try:
            from clefts.libs.msentity.msentity import MSDataset
        except ImportError:
            from msentity import MSDataset
        return MSDataset.load(str(path), load_peak_metadata=False).metadata
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
        if isinstance(value, dict):
            for key in ("rows", "records", "data", "results"):
                if isinstance(value.get(key), list):
                    value = value[key]
                    break
        if not isinstance(value, list):
            raise ValueError("JSON input must be an array, or contain a rows/records/data/results array.")
        return pd.DataFrame(value)
    raise ValueError(f"Unsupported table type {suffix!r}.")


def _similarity_table(path: Path) -> pd.DataFrame:
    """Load the public msentity SimilarityDataset schema without requiring its version."""
    with h5py.File(path, "r") as handle:
        file_format = handle.attrs.get("format")
        if isinstance(file_format, bytes):
            file_format = file_format.decode("utf-8")
        if file_format != "msentity.similarity":
            raise ValueError("Not an msentity similarity file.")
        if int(handle.attrs.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported msentity similarity schema version.")
        if "table.parquet" not in handle:
            raise ValueError("Similarity file does not contain table.parquet.")
        frame = pd.read_parquet(io.BytesIO(handle["table.parquet"][()].tobytes()))
    required = {"index1", "index2", "cosine_similarity"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Similarity table is missing required columns: {sorted(missing)}")
    return frame


def _attach_metadata(
    frame: pd.DataFrame, metadata_path: str, join_column: str | None
) -> tuple[pd.DataFrame, str]:
    path = Path(metadata_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Metadata file does not exist: {path}")
    metadata = _plain_table(path).reset_index(drop=True)
    resolved = join_column or ("SpecID" if "SpecID" in frame.columns and "SpecID" in metadata.columns else None)
    if not resolved:
        common = [str(column) for column in frame.columns if column in metadata.columns]
        raise ValueError(
            "Select a join column shared by the .mssim result and metadata file. "
            f"Common columns: {common or 'none'}"
        )
    if resolved not in frame.columns:
        raise ValueError(f"Join column {resolved!r} was not found in the .mssim result.")
    if resolved not in metadata.columns:
        raise ValueError(f"Join column {resolved!r} was not found in metadata file {path}.")
    non_missing = metadata[resolved].notna()
    duplicate = metadata.loc[non_missing, resolved].duplicated(keep=False)
    if duplicate.any():
        examples = metadata.loc[non_missing].loc[duplicate, resolved].drop_duplicates().tolist()[:5]
        raise ValueError(
            f"Metadata join column {resolved!r} must be unique; duplicate examples: {examples}"
        )
    renamed = {
        column: f"metadata.{column}"
        for column in metadata.columns
        if column != resolved and column in frame.columns
    }
    metadata = metadata.rename(columns=renamed)
    combined = frame.merge(metadata, on=resolved, how="left", sort=False, validate="many_to_one")
    return combined, resolved


def read_table(
    source: str | Path, *, metadata: str | None = None, join_column: str | None = None
) -> pd.DataFrame:
    """Read a supported tabular result file without changing its columns."""
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Input file does not exist: {path}")
    if path.suffix.lower() == ".mssim":
        frame = _similarity_table(path)
        if metadata:
            frame, _ = _attach_metadata(frame, metadata, join_column)
        return frame
    if metadata:
        raise ValueError("Metadata sources can only be attached to an .mssim input.")
    try:
        return _plain_table(path)
    except ValueError as error:
        raise ValueError(f"Unsupported input type {path.suffix.lower()!r}. Supported: {', '.join(SUPPORTED_SUFFIXES)}") from error
