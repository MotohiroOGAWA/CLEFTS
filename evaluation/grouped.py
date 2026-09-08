from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import read_table


def _validate_items(items: Any, label: str) -> list[dict[str, str]]:
    if not isinstance(items, list) or not items:
        raise ValueError(f"At least one {label} is required.")
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Each {label} must be an object.")
        item_id, name = str(item.get("id", "")).strip(), str(item.get("name", "")).strip()
        if not item_id or not name:
            raise ValueError(f"Each {label} requires a non-empty id and name.")
        normalized.append({"id": item_id, "name": name})
    ids = [item["id"] for item in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label.capitalize()} ids must be unique.")
    return normalized


def _series(items: Any) -> list[dict[str, str]]:
    normalized = _validate_items(items, "series")
    source = {str(item.get("id", "")): item for item in items}
    for item in normalized:
        color = str(source[item["id"]].get("color", "#36c5a2"))
        if len(color) != 7 or not color.startswith("#"):
            raise ValueError(f"Invalid series color: {color}")
        try:
            int(color[1:], 16)
        except ValueError as error:
            raise ValueError(f"Invalid series color: {color}") from error
        item["color"] = color
    return normalized


def _statistics(values: pd.Series, outlier_limit: int = 250) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric[np.isfinite(numeric) & numeric.between(0, 1)].sort_values()
    invalid_count = int(len(values) - len(valid))
    if valid.empty:
        return {
            "count": 0, "invalidCount": invalid_count, "min": None, "q1": None,
            "median": None, "q3": None, "max": None, "whiskerLow": None,
            "whiskerHigh": None, "outliers": [], "outlierCount": 0,
        }
    q1, median, q3 = (float(valid.quantile(q)) for q in (0.25, 0.5, 0.75))
    iqr = q3 - q1
    central = valid[(valid >= q1 - 1.5 * iqr) & (valid <= q3 + 1.5 * iqr)]
    outliers = valid[~valid.index.isin(central.index)].tolist()
    if len(outliers) > outlier_limit:
        indices = np.linspace(0, len(outliers) - 1, outlier_limit, dtype=int)
        displayed = [float(outliers[index]) for index in indices]
    else:
        displayed = [float(value) for value in outliers]
    return {
        "count": int(len(valid)), "invalidCount": invalid_count,
        "min": float(valid.min()), "q1": q1, "median": median, "q3": q3,
        "max": float(valid.max()), "whiskerLow": float(central.min()),
        "whiskerHigh": float(central.max()), "outliers": displayed,
        "outlierCount": len(outliers),
    }


def compare(config: dict[str, Any]) -> dict[str, Any]:
    """Compare one similarity result for every configured group/series pair."""
    if not isinstance(config, dict):
        raise ValueError("Grouped evaluation configuration must be an object.")
    groups = _validate_items(config.get("groups"), "group")
    series = _series(config.get("series"))
    group_ids, series_ids = {item["id"] for item in groups}, {item["id"] for item in series}
    entries = config.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("entries must be an array.")
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each result entry must be an object.")
        pair = (str(entry.get("groupId", "")), str(entry.get("seriesId", "")))
        if pair[0] not in group_ids or pair[1] not in series_ids:
            raise ValueError(f"Result entry refers to an unknown group or series: {pair}")
        if pair in by_pair:
            raise ValueError(f"Only one result can be assigned to group/series pair {pair}.")
        by_pair[pair] = entry

    cells = []
    for group in groups:
        for item in series:
            entry = by_pair.get((group["id"], item["id"]), {})
            no_data = bool(entry.get("noData", not entry.get("path")))
            cell = {
                "groupId": group["id"], "seriesId": item["id"],
                "path": str(entry.get("path", "")),
            }
            if no_data:
                cell.update(_statistics(pd.Series([], dtype=float)))
                cell["status"] = "no-data"
            else:
                path = Path(cell["path"]).expanduser().resolve()
                if path.suffix.lower() != ".mssim":
                    raise ValueError(f"Grouped box-plot input must be an .mssim file: {path}")
                frame = read_table(path)
                cell.update(_statistics(frame["cosine_similarity"]))
                cell["sourceRows"] = int(len(frame))
                cell["status"] = "ok" if cell["count"] else "no-data"
            cells.append(cell)
    return {
        "title": str(config.get("title", "Similarity comparison")).strip() or "Similarity comparison",
        "groups": groups, "series": series, "cells": cells,
    }


def grouped_box_plot_svg(
    result: dict[str, Any], *, width: int = 1000, height: int = 600,
    transparent: bool = True, background_color: str = "#ffffff",
    text_color: str = "#555555", group_gap: float = 56.0,
    series_gap: float = 6.0, box_width: float = 0.0,
) -> str:
    if width < 400 or height < 300 or width > 8192 or height > 8192:
        raise ValueError("Image size must be between 400 x 300 and 8192 x 8192 pixels.")
    groups, series, cells = result["groups"], result["series"], result["cells"]
    if group_gap < 0 or series_gap < 0 or box_width < 0:
        raise ValueError("Group gap, series gap, and box width must not be negative.")
    cell_by_pair = {(cell["groupId"], cell["seriesId"]): cell for cell in cells}
    margin_left, margin_right, margin_top, margin_bottom = 70, 28, 85, 82
    plot_width, plot_height = width - margin_left - margin_right, height - margin_top - margin_bottom
    if len(groups) > 1:
        maximum_group_gap = max(0.0, (plot_width - len(groups) * 20) / (len(groups) - 1))
        resolved_group_gap = min(group_gap, maximum_group_gap)
    else:
        resolved_group_gap = 0.0
    group_width = (plot_width - resolved_group_gap * (len(groups) - 1)) / len(groups)
    if len(series) > 1:
        maximum_series_gap = max(0.0, (group_width - len(series) * 3) / (len(series) - 1))
        resolved_series_gap = min(series_gap, maximum_series_gap)
    else:
        resolved_series_gap = 0.0
    maximum_box_width = max(1.0, (group_width - resolved_series_gap * (len(series) - 1)) / len(series))
    resolved_box_width = maximum_box_width if box_width == 0 else min(box_width, maximum_box_width)
    cluster_width = resolved_box_width * len(series) + resolved_series_gap * (len(series) - 1)
    y = lambda value: margin_top + (1 - max(0.0, min(1.0, float(value)))) * plot_height
    marks = []
    for tick in (0, .25, .5, .75, 1):
        tick_y = y(tick)
        marks.append(f'<path class="grid" d="M{margin_left} {tick_y:.2f}H{margin_left + plot_width}"/><text x="{margin_left - 9}" y="{tick_y + 4:.2f}" text-anchor="end" class="label">{tick:g}</text>')
    for group_index, group in enumerate(groups):
        group_start = margin_left + group_index * (group_width + resolved_group_gap)
        center = group_start + group_width / 2
        marks.append(f'<text x="{center:.2f}" y="{margin_top + plot_height + 25}" text-anchor="middle" class="group-label">{escape(group["name"])}</text>')
        if group_index:
            boundary = group_start - resolved_group_gap / 2
            marks.append(f'<path class="separator" d="M{boundary:.2f} {margin_top}V{margin_top + plot_height}"/>')
        for series_index, item in enumerate(series):
            x = (
                group_start + (group_width - cluster_width) / 2
                + series_index * (resolved_box_width + resolved_series_gap)
                + resolved_box_width / 2
            )
            cell = cell_by_pair[(group["id"], item["id"])]
            color = escape(item["color"])
            if cell["median"] is None:
                marks.append(f'<path class="no-data" d="M{x-5:.2f} {y(.05)-5:.2f}l10 10m0-10l-10 10"/><text x="{x:.2f}" y="{y(.05)-10:.2f}" text-anchor="middle" class="no-data-label">No data</text>')
                continue
            low, q1, median, q3, high = (y(cell[key]) for key in ("whiskerLow", "q1", "median", "q3", "whiskerHigh"))
            left, right = x - resolved_box_width / 2, x + resolved_box_width / 2
            marks.extend([
                f'<path class="whisker" d="M{x:.2f} {low:.2f}V{q1:.2f}M{x:.2f} {q3:.2f}V{high:.2f}M{left:.2f} {low:.2f}H{right:.2f}M{left:.2f} {high:.2f}H{right:.2f}"/>',
                f'<rect x="{left:.2f}" y="{q3:.2f}" width="{resolved_box_width:.2f}" height="{max(1, q1-q3):.2f}" fill="{color}" fill-opacity=".62" stroke="{color}"/>',
                f'<path class="median" stroke="{color}" d="M{left:.2f} {median:.2f}H{right:.2f}"/>',
                *[f'<circle cx="{x:.2f}" cy="{y(value):.2f}" r="2.2" fill="{color}"/>' for value in cell["outliers"]],
            ])
    legend_width = plot_width / max(1, len(series))
    legend = "".join(
        f'<rect x="{margin_left + index * legend_width:.2f}" y="48" width="14" height="10" fill="{escape(item["color"])}"/><text x="{margin_left + index * legend_width + 20:.2f}" y="58" class="legend">{escape(item["name"])}</text>'
        for index, item in enumerate(series)
    )
    background = "" if transparent else f'<rect width="{width}" height="{height}" fill="{escape(background_color)}"/>'
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<style>text{{font-family:system-ui,-apple-system,sans-serif;fill:{escape(text_color)}}}.title{{font-size:18px;font-weight:600}}.label,.legend{{font-size:11px}}.group-label{{font-size:12px;font-weight:600}}.axis{{stroke:{escape(text_color)}}}.grid{{stroke:{escape(text_color)};stroke-opacity:.2}}.separator{{stroke:{escape(text_color)};stroke-opacity:.12}}.whisker{{stroke:{escape(text_color)};stroke-width:1.4}}.median{{stroke-width:3}}.no-data{{stroke:{escape(text_color)};stroke-width:1.5;stroke-opacity:.65}}.no-data-label{{font-size:9px;fill:{escape(text_color)};fill-opacity:.75}}</style>'
        f'{background}<text x="{margin_left}" y="27" class="title">{escape(result["title"])}</text>{legend}'
        f'<path class="axis" fill="none" d="M{margin_left} {margin_top}V{margin_top + plot_height}H{margin_left + plot_width}"/>'
        f'<text transform="translate(18,{margin_top + plot_height / 2}) rotate(-90)" text-anchor="middle" class="group-label">Cosine similarity</text>'
        f'{"".join(marks)}</svg>'
    )
