from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .chart import box_plot_svg, save_svg
from .core import EvaluationRequest, inspect_dataset, summarize
from .grouped import compare, grouped_box_plot_svg


def _json_list(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array.")
    return tuple(str(item) for item in parsed)


def _bins(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _grouped_config(args: argparse.Namespace) -> dict:
    if args.request_json:
        value = json.loads(args.request_json)
    else:
        with open(args.config, encoding="utf-8") as stream:
            value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("Grouped evaluation configuration must be a JSON object.")
    if value.get("schema") == "clefts.evaluation.config":
        if value.get("schemaVersion") != 1 or value.get("kind") != "grouped":
            raise ValueError("Expected a version 1 grouped evaluation configuration.")
        value = value.get("config")
        if not isinstance(value, dict):
            raise ValueError("Grouped evaluation configuration payload is missing.")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evaluation", description="Group and summarize CLEFTS result tables.")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Inspect input columns and inferred types.")
    inspect.add_argument("--input", required=True)
    inspect.add_argument("--metadata")
    inspect.add_argument("--join-column")
    summary = commands.add_parser("summarize", help="Create grouped cosine-similarity box-plot data.")
    summary.add_argument("--input", required=True)
    summary.add_argument("--metadata")
    summary.add_argument("--join-column")
    summary.add_argument("--group-column", required=True)
    summary.add_argument("--mode", choices=("auto", "categorical", "numeric"), default="auto")
    summary.add_argument("--bins", default="", help="Comma-separated boundaries, e.g. 0,10,20.")
    summary.add_argument("--include", help="JSON array of category/range labels to include.")
    summary.add_argument("--order", help="JSON array defining output order.")
    summary.add_argument("--width", type=int, default=900)
    summary.add_argument("--height", type=int, default=520)
    summary.add_argument("--color", default="#36c5a2")
    summary.add_argument("--opaque", action="store_true", help="Use a white background instead of transparency.")
    summary.add_argument("--output-image", help="Optional SVG output path.")
    summary.add_argument("--output-json", help="Optional JSON result path.")
    grouped = commands.add_parser("compare", help="Compare multiple .mssim files by group and series.")
    source = grouped.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="Grouped evaluation JSON configuration path.")
    source.add_argument("--request-json", help="Grouped evaluation configuration as JSON.")
    grouped.add_argument("--width", type=int)
    grouped.add_argument("--height", type=int)
    grouped.add_argument("--opaque", action="store_true", help="Use a solid background instead of transparency.")
    grouped.add_argument("--background-color")
    grouped.add_argument("--text-color")
    grouped.add_argument("--output-image", help="Optional SVG output path.")
    grouped.add_argument("--output-json", help="Optional JSON result path.")
    return parser


def execute(args: argparse.Namespace) -> dict:
    if args.command == "inspect":
        return inspect_dataset(args.input, args.metadata, args.join_column)
    if args.command == "compare":
        config = _grouped_config(args)
        result = compare(config)
        result["svg"] = grouped_box_plot_svg(
            result,
            width=args.width or int(config.get("width", 1000)),
            height=args.height or int(config.get("height", 600)),
            transparent=not args.opaque and bool(config.get("transparent", True)),
            background_color=args.background_color or str(config.get("backgroundColor", "#ffffff")),
            text_color=args.text_color or str(config.get("textColor", "#555555")),
            group_gap=float(config.get("groupGap", 56)),
            series_gap=float(config.get("seriesGap", 6)),
            box_width=float(config.get("boxWidth", 0)),
        )
        if args.output_image:
            save_svg(result["svg"], args.output_image)
            result["imagePath"] = args.output_image
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as stream:
                json.dump({key: value for key, value in result.items() if key != "svg"}, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        return result
    request = EvaluationRequest(
        input_path=args.input, group_column=args.group_column,
        metadata=args.metadata, join_column=args.join_column, mode=args.mode,
        bins=_bins(args.bins), include=_json_list(args.include), order=_json_list(args.order) or (),
    )
    result = summarize(request)
    result["svg"] = box_plot_svg(result, width=args.width, height=args.height, color=args.color, transparent=not args.opaque)
    if args.output_image:
        save_svg(result["svg"], args.output_image)
        result["imagePath"] = args.output_image
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as stream:
            json.dump({key: value for key, value in result.items() if key != "svg"}, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    try:
        result = execute(build_parser().parse_args(argv))
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
