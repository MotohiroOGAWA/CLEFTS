from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

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


def _categorical_groups(series: pd.Series) -> pd.Series:
    return series.map(lambda value: MISSING_LABEL if pd.isna(value) else str(value))


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
    detected_numeric = pd.api.types.is_numeric_dtype(frame[request.group_column])
    mode = request.mode if request.mode != "auto" else ("numeric" if detected_numeric else "categorical")
    if mode not in {"categorical", "numeric"}:
        raise ValueError("Grouping mode must be auto, categorical, or numeric.")
    group_key = _numeric_groups(frame[request.group_column], request.bins) if mode == "numeric" else _categorical_groups(frame[request.group_column])
    rows = _boxplot_rows(frame, group_key)
    if request.include is not None:
        included = set(request.include)
        rows = [row for row in rows if row["category"] in included]
    if request.order:
        positions = {name: index for index, name in enumerate(request.order)}
        rows.sort(key=lambda row: (positions.get(row["category"], len(positions)), row["category"]))
    return {
        "input": request.input_path,
        "groupColumn": request.group_column,
        "groupMode": mode,
        "scoreColumn": "cosine_similarity",
        "sourceRows": int(len(frame)),
        "rows": rows,
    }
