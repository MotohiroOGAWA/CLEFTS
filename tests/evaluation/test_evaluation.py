from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from evaluation.chart import box_plot_svg
from evaluation.cli import build_parser, execute
from evaluation.core import EvaluationRequest, inspect_dataset, summarize


def _write_similarity(path, table):
    payload = table.to_parquet(index=False)
    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "msentity.similarity"
        handle.attrs["schema_version"] = 1
        handle.create_dataset("table.parquet", data=np.frombuffer(payload, dtype=np.uint8))
        handle.create_dataset(
            "metadata.json", data=json.dumps({"row_count": len(table)}),
            dtype=h5py.string_dtype("utf-8"),
        )
    return path


def _fixtures(tmp_path):
    spec_ids = [f"S{index}" for index in range(6)]
    similarity = _write_similarity(
        tmp_path / "scores.mssim",
        pd.DataFrame({
            "SpecID": spec_ids, "index1": range(6), "index2": range(6),
            "cosine_similarity": [0.1, 0.2, 0.3, 0.4, 1.0, 0.5],
        }),
    )
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({
        "SpecID": spec_ids,
        "AdductType": ["[M+H]+"] * 5 + ["[M+Na]+"],
        "CollisionEnergy": [5, 7, 12, 15, 25, 25],
    }).to_csv(metadata, index=False)
    return similarity, metadata


def test_default_spec_id_join_and_categorical_boxplot(tmp_path):
    similarity, source = _fixtures(tmp_path)
    inspected = inspect_dataset(str(similarity), str(source))
    names = {column["name"] for column in inspected["columns"]}
    assert {"SpecID", "cosine_similarity", "AdductType", "CollisionEnergy"} <= names

    result = summarize(EvaluationRequest(
        str(similarity), "AdductType", metadata=str(source), include=("[M+H]+",)
    ))
    assert len(result["rows"]) == 1
    box = result["rows"][0]
    assert box["category"] == "[M+H]+"
    assert box["count"] == 5
    assert box["median"] == 0.3
    assert box["whiskerHigh"] == 0.4
    assert box["outliers"] == [1.0]
    assert summarize(EvaluationRequest(
        str(similarity), "AdductType", metadata=str(source), include=()
    ))["rows"] == []


def test_numeric_ranges_are_grouped_and_ordered(tmp_path):
    similarity, source = _fixtures(tmp_path)
    result = summarize(EvaluationRequest(
        str(similarity), "CollisionEnergy", metadata=str(source), mode="numeric",
        bins=(0, 10, 20), order=("[20,~)", "[0,10)", "[10,20)"),
    ))
    assert [row["category"] for row in result["rows"]] == ["[20,~)", "[0,10)", "[10,20)"]
    assert [row["count"] for row in result["rows"]] == [2, 2, 2]


def test_cli_writes_transparent_colored_boxplot(tmp_path):
    similarity, source = _fixtures(tmp_path)
    image = tmp_path / "evaluation.svg"
    args = build_parser().parse_args([
        "summarize", "--input", str(similarity), "--metadata", str(source),
        "--group-column", "AdductType", "--include", json.dumps(["[M+Na]+"]),
        "--width", "500", "--height", "300", "--color", "#ff00aa",
        "--output-image", str(image),
    ])
    result = execute(args)
    svg = image.read_text(encoding="utf-8")
    assert result["rows"][0]["category"] == "[M+Na]+"
    assert 'width="500" height="300"' in svg
    assert 'fill="#ff00aa"' in svg
    assert '<rect width="500" height="300" fill="#ffffff"/>' not in svg
    assert box_plot_svg(result, transparent=False).count('fill="#ffffff"') == 1


def test_explicit_join_column_supports_msds_metadata(tmp_path):
    app_root = Path(__file__).parents[2]
    msds = app_root / "clefts/libs/msentity/tests/test_data/sample.msds"
    identifiers = ["MassSpecID0000001", "MassSpecID0000002"]
    similarity = _write_similarity(
        tmp_path / "scores.mssim",
        pd.DataFrame({
            "IDENTIFIER": identifiers, "index1": [0, 1], "index2": [0, 1],
            "cosine_similarity": [0.9, 0.5],
        }),
    )
    inspected = inspect_dataset(str(similarity), str(msds), "IDENTIFIER")
    names = {column["name"] for column in inspected["columns"]}
    assert {"ADDUCT", "CollisionEnergy", "cosine_similarity"} <= names
    result = summarize(EvaluationRequest(
        str(similarity), "CollisionEnergy", metadata=str(msds),
        join_column="IDENTIFIER", mode="numeric", bins=(0, 25, 35),
    ))
    assert [(row["category"], row["count"]) for row in result["rows"]] == [
        ("[25,35)", 1), ("[0,25)", 1)
    ]
