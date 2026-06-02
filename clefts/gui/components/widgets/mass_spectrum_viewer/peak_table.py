from __future__ import annotations

import pandas as pd


def make_peak_dataframe(
    mz: list[float],
    intensity: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "peak_id": list(range(len(mz))),
            "mz": [float(value) for value in mz],
            "intensity": [float(value) for value in intensity],
        }
    )


def make_peak_table(
    peaks: pd.DataFrame,
    *,
    selected_peak_id: int | None = None,
    hovered_peak_id: int | None = None,
):
    table = peaks.copy()

    def highlight(row):
        peak_id = int(row["peak_id"])

        if selected_peak_id == peak_id:
            return [
                "background-color: #ffe0e0; font-weight: bold;"
            ] * len(row)

        if hovered_peak_id == peak_id:
            return [
                "background-color: #fff2cc; font-weight: bold;"
            ] * len(row)

        return [""] * len(row)

    return table.style.apply(highlight, axis=1)


def extract_peak_id_from_table_event(
    peaks: pd.DataFrame,
    evt,
) -> int | None:
    if evt is None:
        return None

    index = getattr(evt, "index", None)

    if index is None:
        return None

    if isinstance(index, list | tuple):
        row_index = index[0]
    else:
        row_index = index

    if row_index is None:
        return None

    return int(peaks.iloc[int(row_index)]["peak_id"])