from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem

from clefts.domain.mass.parse_ce import parse_ce_to_ev
from clefts.domain.molecule.descriptors import (
    DEFAULT_DESCRIPTOR_NAMES,
    compute_descriptor_values,
)

from .io import read_table


MISSING_LABEL = "(missing)"


@dataclass(frozen=True)
class EvaluationRequest:
    input_path: str
    group_column: str
    metadata: str | None = None
    join_column: str | None = None
    mode: str = "auto"
    bins: tuple[float, ...] = ()
    include: tuple[str, ...] | None = None
    order: tuple[str, ...] = ()
    transform: str = "none"
    precursor_mz_column: str = "PrecursorMZ"
    instrument_column: str | None = None
    smiles_column: str = "SMILES"
    chemical_descriptor: str = "HeavyAtomCount"


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def inspect_dataset(
    input_path: str, metadata: str | None = None, join_column: str | None = None
) -> dict[str, Any]:
    frame = read_table(input_path, metadata=metadata, join_column=join_column)
    columns = []
    for name in frame.columns:
        series = frame[name]
        numeric = pd.to_numeric(series, errors="coerce")
        non_missing = int(series.notna().sum())
        numeric_count = int(numeric.notna().sum())
        is_numeric = non_missing > 0 and numeric_count == non_missing
        entry: dict[str, Any] = {
            "name": str(name),
            "kind": "numeric" if is_numeric else "categorical",
            "nonMissing": non_missing,
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
        }
        if is_numeric:
            entry["min"] = _json_value(numeric.min())
            entry["max"] = _json_value(numeric.max())
        columns.append(entry)
    return {
        "path": input_path, "metadata": metadata, "joinColumn": join_column,
        "rows": int(len(frame)), "columns": columns,
    }


def _number_label(value: float) -> str:
    return f"{value:g}"


def _numeric_groups(series: pd.Series, edges: Iterable[float]) -> pd.Series:
    cuts = sorted(set(float(value) for value in edges))
    if not cuts:
        raise ValueError("Numeric grouping requires at least one bin boundary.")
    if not all(np.isfinite(cuts)):
        raise ValueError("Bin boundaries must be finite numbers.")
    values = pd.to_numeric(series, errors="coerce")
    bounds = [-np.inf, *cuts, np.inf]
    labels = [f"[-~,{_number_label(cuts[0])})"]
    labels.extend(
        f"[{_number_label(left)},{_number_label(right)})"
        for left, right in zip(cuts, cuts[1:])
    )
    labels.append(f"[{_number_label(cuts[-1])},~)")
    grouped = pd.cut(values, bins=bounds, labels=labels, right=False, ordered=True)
    return grouped.astype(object).where(values.notna(), MISSING_LABEL)


def _numeric_group_order(edges: Iterable[float]) -> list[str]:
    cuts = sorted(set(float(value) for value in edges))
    if not cuts:
        return [MISSING_LABEL]
    return [
        f"[-~,{_number_label(cuts[0])})",
        *(
            f"[{_number_label(left)},{_number_label(right)})"
            for left, right in zip(cuts, cuts[1:])
        ),
        f"[{_number_label(cuts[-1])},~)",
        MISSING_LABEL,
    ]


def _categorical_groups(series: pd.Series) -> pd.Series:
    return series.map(lambda value: MISSING_LABEL if pd.isna(value) else str(value))


def _collision_energy_values(frame: pd.DataFrame, request: EvaluationRequest) -> pd.Series:
    precursor = (
        frame[request.precursor_mz_column]
        if request.precursor_mz_column in frame.columns
        else pd.Series([None] * len(frame), index=frame.index)
    )
    instrument = (
        frame[request.instrument_column]
        if request.instrument_column and request.instrument_column in frame.columns
        else pd.Series([None] * len(frame), index=frame.index)
    )
    return pd.Series([
        parse_ce_to_ev(ce, mz, inst)
        for ce, mz, inst in zip(frame[request.group_column], precursor, instrument)
    ], index=frame.index, dtype=float)


def _chemical_descriptor_values(
    frame: pd.DataFrame, request: EvaluationRequest
) -> pd.Series:
    if request.smiles_column not in frame.columns:
        raise ValueError(f"SMILES column was not found: {request.smiles_column}")
    if request.chemical_descriptor not in DEFAULT_DESCRIPTOR_NAMES:
        raise ValueError(
            f"Unsupported chemical descriptor: {request.chemical_descriptor}"
        )
    cache: dict[str, float | None] = {}
    values = []
    for raw_smiles in frame[request.smiles_column]:
        smiles = "" if pd.isna(raw_smiles) else str(raw_smiles)
        if smiles not in cache:
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            cache[smiles] = (
                compute_descriptor_values(mol, (request.chemical_descriptor,))[0]
                if mol is not None else None
            )
        values.append(cache[smiles])
    return pd.Series(values, index=frame.index, dtype=float)


def _boxplot_rows(frame: pd.DataFrame, group_key: pd.Series) -> list[dict[str, Any]]:
    if "cosine_similarity" not in frame.columns:
        raise ValueError("Input must contain the .mssim cosine_similarity column.")
    values = pd.to_numeric(frame["cosine_similarity"], errors="coerce")
    grouped = pd.DataFrame({"__group": group_key, "__value": values}).groupby(
        "__group", sort=False, dropna=False
    )
    rows = []
    for category, group in grouped:
        scores = group["__value"].dropna().sort_values()
        if scores.empty:
            rows.append({"category": str(category), "count": 0, "min": None, "q1": None,
                         "median": None, "q3": None, "max": None,
                         "whiskerLow": None, "whiskerHigh": None, "outliers": []})
            continue
        q1, median, q3 = (float(scores.quantile(q)) for q in (0.25, 0.5, 0.75))
        iqr = q3 - q1
        central = scores[(scores >= q1 - 1.5 * iqr) & (scores <= q3 + 1.5 * iqr)]
        outliers = scores[~scores.index.isin(central.index)].tolist()
        rows.append({
            "category": str(category), "count": int(len(scores)),
            "min": float(scores.min()), "q1": q1, "median": median, "q3": q3,
            "max": float(scores.max()), "whiskerLow": float(central.min()),
            "whiskerHigh": float(central.max()), "outliers": [float(value) for value in outliers],
        })
    return rows


def summarize(request: EvaluationRequest) -> dict[str, Any]:
    frame = read_table(
        request.input_path, metadata=request.metadata, join_column=request.join_column
    )
    if request.group_column not in frame.columns:
        raise ValueError(f"Group column was not found: {request.group_column}")
    if request.transform == "collision-energy":
        source = _collision_energy_values(frame, request)
        effective_column = f"{request.group_column} (eV)"
    elif request.transform == "chemical":
        source = _chemical_descriptor_values(frame, request)
        effective_column = request.chemical_descriptor
    elif request.transform == "none":
        source = frame[request.group_column]
        effective_column = request.group_column
    else:
        raise ValueError("Transform must be none, collision-energy, or chemical.")
    detected_numeric = pd.api.types.is_numeric_dtype(source)
    mode = request.mode if request.mode != "auto" else ("numeric" if detected_numeric else "categorical")
    if mode not in {"categorical", "numeric"}:
        raise ValueError("Grouping mode must be auto, categorical, or numeric.")
    group_key = _numeric_groups(source, request.bins) if mode == "numeric" else _categorical_groups(source)
    rows = _boxplot_rows(frame, group_key)
    if request.include is not None:
        included = set(request.include)
        rows = [row for row in rows if row["category"] in included]
    if request.order:
        positions = {name: index for index, name in enumerate(request.order)}
        rows.sort(key=lambda row: (positions.get(row["category"], len(positions)), row["category"]))
    elif mode == "numeric":
        numeric_order = {
            name: index for index, name in enumerate(_numeric_group_order(request.bins))
        }
        rows.sort(key=lambda row: numeric_order.get(row["category"], len(numeric_order)))
    return {
        "input": request.input_path,
        "groupColumn": effective_column,
        "sourceGroupColumn": request.group_column,
        "transform": request.transform,
        "groupMode": mode,
        "scoreColumn": "cosine_similarity",
        "sourceRows": int(len(frame)),
        "rows": rows,
    }
