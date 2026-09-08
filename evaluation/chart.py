from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any


def box_plot_svg(
    result: dict[str, Any], *, width: int = 900, height: int = 520,
    color: str = "#36c5a2", transparent: bool = True,
) -> str:
    if width < 240 or height < 200:
        raise ValueError("Image size must be at least 240 x 200 pixels.")
    rows = result["rows"]
    margin_left, margin_right, margin_top, margin_bottom = 62, 24, 45, 100
    plot_width = max(1, width - margin_left - margin_right)
    plot_height = max(1, height - margin_top - margin_bottom)
    slot = plot_width / max(1, len(rows))
    box_width = max(8.0, min(70.0, slot * 0.55))
    background = "" if transparent else f'<rect width="{width}" height="{height}" fill="#ffffff"/>'
    marks = [f'<path class="grid" d="M{margin_left} {margin_top + plot_height * (1-tick):.2f}H{margin_left + plot_width}"/><text x="{margin_left - 9}" y="{margin_top + plot_height * (1-tick) + 4:.2f}" text-anchor="end" class="label">{tick:g}</text>' for tick in (0, .25, .5, .75, 1)]
    y = lambda value: margin_top + (1 - max(0.0, min(1.0, float(value)))) * plot_height
    for index, row in enumerate(rows):
        x = margin_left + index * slot + slot / 2
        label = escape(str(row["category"]))
        if row["median"] is not None:
            low, q1, median, q3, high = (y(row[key]) for key in ("whiskerLow", "q1", "median", "q3", "whiskerHigh"))
            left, right = x - box_width / 2, x + box_width / 2
            marks.extend([
                f'<path class="whisker" d="M{x:.2f} {low:.2f}V{q1:.2f}M{x:.2f} {q3:.2f}V{high:.2f}M{left:.2f} {low:.2f}H{right:.2f}M{left:.2f} {high:.2f}H{right:.2f}"/>',
                f'<rect x="{left:.2f}" y="{q3:.2f}" width="{box_width:.2f}" height="{max(1, q1-q3):.2f}" fill="{escape(color)}" fill-opacity=".55" stroke="{escape(color)}"/>',
                f'<path class="median" stroke="{escape(color)}" d="M{left:.2f} {median:.2f}H{right:.2f}"/>',
                *[f'<circle cx="{x:.2f}" cy="{y(value):.2f}" r="3" fill="{escape(color)}"/>' for value in row["outliers"]],
            ])
        marks.append(f'<text transform="translate({x:.2f},{margin_top + plot_height + 13}) rotate(35)" text-anchor="start" class="label">{label}</text>')
    title = escape(f'Cosine similarity by {result["groupColumn"]}')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<style>text{font-family:system-ui,-apple-system,sans-serif;fill:#555}.title{font-size:17px;font-weight:600}.label{font-size:11px}.axis{stroke:#777;stroke-width:1}.grid{stroke:#999;stroke-opacity:.2}.whisker{stroke:#666;stroke-width:1.5}.median{stroke-width:3}</style>'
        f'{background}<text x="{margin_left}" y="25" class="title">{title}</text>'
        f'<path class="axis" fill="none" d="M{margin_left} {margin_top}V{margin_top + plot_height}H{margin_left + plot_width}"/>'
        f'{"".join(marks)}</svg>'
    )


def save_svg(svg: str, target: str | Path) -> None:
    path = Path(target).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
