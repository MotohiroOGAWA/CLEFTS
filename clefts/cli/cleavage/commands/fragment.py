from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from clefts.cli.base import CLICommand
from clefts.domain.fragment.cleavage.CleavagePatternSet import CleavagePatternSet
from clefts.domain.fragment.cleavage.CleavageResult import CleavageResult
from clefts.libs.mmkit.mmkit import Compound


class FragmentCommand(CLICommand):
    name = "fragment"
    help = "Fragment a SMILES using a CleavagePatternSet JSON file."
    description = (
        "Apply all cleavage patterns in a CleavagePatternSet JSON file "
        "to one input SMILES and print fragment products as JSON."
    )
    aliases = ("frag",)
    order = 10

    def configure(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        parser.add_argument(
            "--smiles",
            required=True,
            help="Input molecule SMILES.",
        )
        parser.add_argument(
            "--patterns",
            type=Path,
            required=True,
            help="CleavagePatternSet JSON file.",
        )
        parser.add_argument(
            "--pretty",
            action="store_true",
            help="Pretty-print JSON output.",
        )

    def run(
        self,
        args: argparse.Namespace,
    ) -> None:
        compound = Compound.from_smiles(args.smiles)
        pattern_set = load_pattern_set(args.patterns)

        cleavage_results = pattern_set.fragment_all(compound)

        output_data = {
            "summary": {
                "input_smiles": args.smiles,
                "canonical_smiles": compound.smiles,
                "pattern_file": str(args.patterns),
                "pattern_set_name": pattern_set.name,
                "num_patterns": len(pattern_set.patterns),
                "num_matched_patterns": len(cleavage_results),
                "num_products": sum(
                    len(result.products)
                    for result in cleavage_results
                ),
            },
            "results": [
                cleavage_result_to_dict(
                    result,
                    cleavage_pattern_id=pattern_set.get_id(result.cleavage),
                )
                for result in cleavage_results
            ],
        }

        print_json(
            data=output_data,
            pretty=args.pretty,
        )


def load_pattern_set(
    path: Path,
) -> CleavagePatternSet:
    data = json.loads(path.read_text(encoding="utf-8"))
    return CleavagePatternSet.from_dict(data)


def cleavage_result_to_dict(
    result: CleavageResult,
    *,
    cleavage_pattern_id: int,
) -> dict[str, Any]:
    return {
        "cleavage_pattern_id": cleavage_pattern_id,
        "cleavage": result.cleavage.to_dict(),
        "reactant_smiles": result.reactant_smiles,
        "products": [
            {
                "smiles": product.smiles,
                "reactant_indices": list(product.reactant_indices),
                "product_indices": list(product.product_indices),
            }
            for product in result.products
        ],
    }


def print_json(
    *,
    data: dict[str, Any],
    pretty: bool,
) -> None:
    print(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2 if pretty else None,
        )
    )