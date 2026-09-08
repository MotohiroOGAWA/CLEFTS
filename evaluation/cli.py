from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .chart import box_plot_svg, save_svg
from .core import EvaluationRequest, inspect_dataset, summarize


def _json_list(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array.")
    return tuple(str(item) for item in parsed)


def _bins(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


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
    return parser


def execute(args: argparse.Namespace) -> dict:
    if args.command == "inspect":
        return inspect_dataset(args.input, args.metadata, args.join_column)
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
